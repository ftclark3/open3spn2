"""Alchemical base transformations for Open3SPN2.

An :class:`AlchemicalTransformation` turns one nucleic base into another through a coupling parameter
lambda, building the single-topology hybrid Hamiltonian

    U(lambda) = (1 - lambda) * U_initial + lambda * U_target.

It builds every identity-dependent DNA force twice -- from the initial DNA and from the target DNA
(``DNA.create_mutant``) -- weighting the two copies with the per-force ``k`` multiplier
(``k_bond``-style global) so their contributions add up to the linear interpolation. Electrostatics is
identity-independent (every DNA base carries charge 0) and is added once. Because ``Exclusion`` encodes
its base-pair exclusions in its energy expression, the two lambda-scaled copies share one identical
exclusion list and coexist on the CPU/GPU platforms -- no special case is needed.
"""
import openmm.unit as unit
import openmm

from .ff3SPN2 import forces


class AlchemicalTransformation:
    """A single-topology transformation of one or more DNA bases into new identities.

    Parameters
    ----------
    dna : DNA
        The initial-state DNA (already built, with its topology computed).
    mutations : iterable
        ``(chainID, resSeq, target)`` tuples (dicts or a DataFrame also work) with ``target`` in
        {A, T, G, C}. The target-state DNA is ``dna.create_mutant(mutations)``.
    lambda_name : str
        Base name for the coupling; the two state weights are ``f'{lambda_name}_initial'`` and
        ``f'{lambda_name}_target'``.
    """

    def __init__(self, dna, mutations, lambda_name='lambda'):
        self.initial = dna
        self.target = dna.create_mutant(mutations)
        self.mutations = mutations
        self.lambda_name = lambda_name

    @property
    def initial_weight(self):
        return f'{self.lambda_name}_initial'

    @property
    def target_weight(self):
        return f'{self.lambda_name}_target'

    def add_forces(self, system, verbose=False):
        """Build the hybrid on `system` (an ``openmm.System`` or ``open3SPN2.System``).

        Adds two lambda-scaled copies of every identity-dependent DNA force -- the initial copy
        (from ``self.initial``, weight ``initial_weight``) and the target copy (from ``self.target``,
        weight ``target_weight``) -- plus a single electrostatics force. Returns
        ``{force_name: initial-state force}`` for per-force-group energy readout."""
        added = {}
        for force_name, ForceClass in forces.items():
            if verbose:
                print(force_name)
            if force_name == 'Electrostatics':                 # identity-independent -> one copy
                force = ForceClass(self.initial)
                force.addForce(system)
                added[force_name] = force
            else:
                initial = ForceClass(self.initial, k=1.0, k_name=self.initial_weight)
                initial.addForce(system)
                target = ForceClass(self.target, k=0.0, k_name=self.target_weight)
                target.addForce(system)
                added[force_name] = initial
        return added

    def set_lambda(self, context, value):
        """Set the coupling on an OpenMM context (0 -> initial identity, 1 -> target identity)."""
        context.setParameter(self.initial_weight, 1.0 - value)
        context.setParameter(self.target_weight, value)

    def get_lambda(self, context):
        """Return the current coupling value from an OpenMM context."""
        return context.getParameter(self.target_weight)

    def energy_derivative(self, context, energy_unit=unit.kilojoule_per_mole):
        """Return dE/dlambda (default kJ/mol) for thermodynamic integration.

        The coupling is linear, so at fixed coordinates dE/dlambda = U(1) - U(0) exactly (independent
        of lambda); evaluating it this way also captures the base-pair / cross-stacking
        ``CustomHbondForce`` contributions that OpenMM cannot expose through energy parameter
        derivatives."""
        saved = (context.getParameter(self.initial_weight), context.getParameter(self.target_weight))
        context.setParameter(self.initial_weight, 1.0)
        context.setParameter(self.target_weight, 0.0)
        u0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
        context.setParameter(self.initial_weight, 0.0)
        context.setParameter(self.target_weight, 1.0)
        u1 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)
        context.setParameter(self.initial_weight, saved[0])
        context.setParameter(self.target_weight, saved[1])
        return u1 - u0


class EnvelopingDistributionSampling:
    """Build a collective-variable force that represents an enveloping distribution over alternative
    DNA sequences.

    This class takes a base DNA object and a list of mutation-sets; each mutation-set is an iterable
    of (chainID, resSeq, target) entries (the same format accepted by DNA.create_mutant). For each
    mutant sequence we construct the usual DNA force objects (with k=1.0) but do not add them
    directly to the System. Instead each OpenMM force object is attached as a collective variable to a
    single openmm.CustomCVForce whose energy is

        U_CV = log( sum_i exp( E_i ) )

    where E_i is the total energy (sum of the identity-dependent DNA terms) of mutant i. The
    CustomCVForce is added to the provided system; the per-mutant DNA forces are not added
    separately.

    Parameters
    ----------
    starting_dna : DNA
        The initial DNA sequence/structure, used to construct the appropriate Force objects for the mutants
    mutant_sets : iterable of iterables
        Each item is an iterable of mutation tuples defining a mutant sequence.
    force_group : int
        Force group to assign to the resulting CustomCVForce (default: 30).
    """

    def __init__(self, starting_dna, mutant_sets, force_group=30):
        self.starting_dna = starting_dna
        # Build mutant DNA objects
        self.mutants = [starting_dna.create_mutant(m) for m in mutant_sets]
        self.mutant_sets = mutant_sets
        self.force_group = force_group
        # Storage for created force wrappers and the collective-variable force
        self._mutant_forces = []  # list of dicts: for each mutant, {force_name: force_wrapper}
        self._cvforce = None

    @staticmethod
    def _collect_force_objects(fwrap):
        """Return list of openmm.Force instances found on a force wrapper object
           (i.e. the multiple Force objects that are associated with the 
           cross stacking and base pairing forces)"""
        found = []
        for v in fwrap.__dict__.values():
            if isinstance(v, openmm.Force):
                found.append(v)
            elif isinstance(v, dict):
                for val in v.values():
                    # tuples/lists of forces
                    if isinstance(val, (tuple, list)):
                        for item in val:
                            if isinstance(item, openmm.Force):
                                found.append(item)
                    elif isinstance(val, openmm.Force):
                        found.append(val)
            elif isinstance(v, (tuple, list)):
                for item in v:
                    if isinstance(item, openmm.Force):
                        found.append(item)
        # fallback: wrapper.force
        if hasattr(fwrap, 'force') and isinstance(getattr(fwrap, 'force'), openmm.Force) and getattr(fwrap, 'force') not in found:
            found.append(getattr(fwrap, 'force'))
        return found

    def add_forces(self, system, verbose=False):
        """Add the DNA forces to the openmm system, 
           putting the sequence-dependent ones in a CustomCVForce
           so that we can construct the Hamiltonian for enveloping distribution sampling.

        Returns a tuple (cvforce, mutant_forces) where mutant_forces is a list of per-mutant dicts.
        """

        # Electrostatics is always sequence-independent, 
        # so we can add it to the openmm system as usual,
        # without regard to the number and kind(s) of mutant sequence(s)
        # that we want to explore with enveloping distribution sampling. 
        # TODO: we should consider removing this block from this class
        #       and instead ask the user to add the sequence-independent
        #       terms to their openmm system outside of this class
        if 'Electrostatics' in forces:
            WrapperClass = forces['Electrostatics']
            elec = WrapperClass(self.starting_dna) # self.starting_dna is the DNA system that was 
                                                   # constructed outside of this class,
                                                   # and whose coordinates will become the 
                                                   # initial configuration for the simulation.                            
            elec.addForce(system)
        else:
            raise ValueError(f"Expected 'Electrostatics' in standard open3SPN2 force dictionary forces, but got {forces}")

        # Initialize our sequence-dependent Open3SPN2 force wrapper classes,
        # assigning each a unique name (a string)
        # and storing the {name: force} pairs in a dictionary.
        sequence_dependent_forces = {}
        for mutant_sequence_index, mutant in enumerate(self.mutants): # loop over lists of mutations
            if verbose:
                print(f"Building forces for mutant {mutant_sequence_index}")
            for force_name, WrapperClass in forces.items(): # ignore sequence-independent_forces
                if force_name == 'Electrostatics':
                    continue
                elif force_name == 'BasePair':
                    open3spn2_force = WrapperClass(mutant)
                    for force_index, openmm_force in open3spn2_force.forces.items():
                        sequence_dependent_forces[f"{force_name}-{force_index}_{mutant_sequence_index}"] = openmm_force
                elif force_name == 'CrossStacking':
                    open3spn2_force = WrapperClass(mutant)
                    for force_index, (openmm_force_c1, openmm_force_c2) in open3spn2_force.crossStackingForces.items():
                        sequence_dependent_forces[f"{force_name}-{force_index}-c1_{mutant_sequence_index}"] = openmm_force_c1
                        sequence_dependent_forces[f"{force_name}-{force_index}-c2_{mutant_sequence_index}"] = openmm_force_c2
                else:
                    sequence_dependent_forces[f"{force_name}_{mutant_sequence_index}"] = WrapperClass(mutant).force

        # set up energy expression to be used for the CustomCVForce
        expr_start = '-0.0083145*300*log('
        expr_end = ')'
        expr_middle = ''
        for mutant_sequence_index in range(len(self.mutants)):
            sub_expr_start = 'exp(-('
            if mutant_sequence_index == len(self.mutants)-1:
                sub_expr_end = ')/(0.0083145*300))'
            else:
                sub_expr_end = ')/(0.0083145*300)+'     
            sub_expr_middle = ''       
            for force_name_index, key in enumerate(sequence_dependent_forces.keys()):
                to_add = ''
                if int(key.split('_')[-1]) == mutant_sequence_index:
                    to_add += key
                if force_name_index == len(sequence_dependent_forces.keys())-1:
                    sub_expr_middle += to_add
                else:
                    sub_expr_middle += f'{to_add}+'
            expr_middle += f'{sub_expr_start}{sub_expr_middle}{sub_expr_end}'
        expr = f'{expr_start}{expr_middle}{expr_end}'
        if verbose:
            print(f'using expression {expr} for enveloping distribution sampling CustomCVForce')

        # initialize and set up CustomCVForce for enveloping distribution sampling
        cv = openmm.CustomCVForce(expr)
        cv.setForceGroup(self.force_group)
        for force_name, force_object in sequence_dependent_forces.items():
            cv.addCollectiveVariable(force_name, force_object) 

        # Add the enveloping distribution sampling CV force to the openmm system
        system.addForce(cv)

        # attach the enveloping distribution sampling CV force to this object for future reference
        self._cvforce = cv

        # not sure if these return values will be useful, can modify in the future if we want
        return cv, self._mutant_forces


    @property 
    def cvforce(self):
        return self._cvforce
    # for convenience and API consistency with AlchemicalTransformation
    def get_cvforce(self):
        return self.cvforce
