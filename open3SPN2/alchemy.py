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
    dna : DNA
        The reference DNA (used to construct mutants).
    mutant_sets : iterable of iterables
        Each item is an iterable of mutation tuples defining a mutant sequence.
    force_group : int
        Force group to assign to the resulting CustomCVForce (default: 30).
    """

    def __init__(self, dna, mutant_sets, force_group=30):
        self.reference = dna
        # Build mutant DNA objects
        self.mutants = [dna.create_mutant(m) for m in mutant_sets]
        self.mutant_sets = mutant_sets
        self.force_group = force_group
        # Storage for created force wrappers and the collective-variable force
        self._mutant_forces = []  # list of dicts: for each mutant, {force_name: force_wrapper}
        self.cvforce = None

    def _collect_force_objects(self, fwrap):
        """Return list of openmm.Force instances found on a force wrapper object."""
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
        """Create per-mutant forces (not added to `system`) and a CustomCVForce that combines them.

        Returns a tuple (cvforce, mutant_forces) where mutant_forces is a list of per-mutant dicts.
        """
        # Create per-mutant force wrapper instances but do not add them to the system.
        self._mutant_forces = []
        for m_idx, mutant in enumerate(self.mutants):
            if verbose:
                print(f"Building forces for mutant {m_idx}")
            mf = {}
            for fname, ForceClass in forces.items():
                # Construct with k=1.0 and ignore k_name
                try:
                    fwrap = ForceClass(mutant, k=1.0)
                except TypeError:
                    # Some force constructors don't accept k; fall back to default
                    fwrap = ForceClass(mutant)
                mf[fname] = fwrap
            self._mutant_forces.append(mf)

        # Add a single Electrostatics force from the reference DNA to the system (option A)
        if 'Electrostatics' in forces:
            ElecClass = forces['Electrostatics']
            elec = ElecClass(self.reference)
            elec.addForce(system)

        # Build the CustomCVForce expression: log( exp(sum_forces_mut0) + exp(sum_forces_mut1) + ... )
        per_mutant_varnames = []
        # We'll also keep a mapping from (m_idx, fname, subidx) -> forceobj for later inspection
        self._cv_force_map = {}
        for m_idx, mf in enumerate(self._mutant_forces):
            varnames = []
            for fname, fwrap in mf.items():
                # Skip Electrostatics (we've added a single copy already)
                if fname == 'Electrostatics':
                    continue
                force_objs = self._collect_force_objects(fwrap)
                if not force_objs:
                    # If no underlying openmm Force found, try adding the wrapper itself if it
                    # behaves like an openmm.Force
                    if isinstance(fwrap, openmm.Force):
                        force_objs = [fwrap]
                for subidx, fobj in enumerate(force_objs):
                    varname = f"{fname}_{m_idx}_{subidx}"
                    varnames.append(varname)
                    self._cv_force_map[(m_idx, fname, subidx)] = fobj
            per_mutant_varnames.append(varnames)

        # per-mutant expression: (fname_0 + fname_1 + ...)
        mutant_exprs = ["(" + "+".join(vs) + ")" if len(vs) > 0 else "(0)" for vs in per_mutant_varnames]
        expr = "log(" + "+".join([f"exp({me})" for me in mutant_exprs]) + ")"

        cv = openmm.CustomCVForce(expr)
        cv.setForceGroup(self.force_group)

        # Add collective variables linking to the per-force OpenMM Force objects
        for (m_idx, fname, subidx), fobj in self._cv_force_map.items():
            varname = f"{fname}_{m_idx}_{subidx}"
            cv.addCollectiveVariable(varname, fobj)

        # Add the combined CV force to the system
        system.addForce(cv)
        self.cvforce = cv
        return cv, self._mutant_forces

    def get_cvforce(self):
        return self.cvforce
