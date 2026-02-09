import numpy as np

import openmm
import openmm.unit as unit
import pandas
from .template import ProteinDNAForce
from .dna import addNonBondedExclusions
_af = 1 * unit.degree / unit.radian  # angle scaling factor
_dnaResidues = ['DA', 'DC', 'DT', 'DG']
_proteinResidues = ['IPR', 'IGL', 'NGP']

class ExclusionProteinDNA(ProteinDNAForce):
    """ Protein-DNA exclusion potential"""
    def __init__(self, dna, protein, k=1, radius_override = None, cutoff = 1.55, force_group=14):
        self.k = k
        self.force_group = force_group
        self.radius_override = radius_override
        self.cutoff = cutoff    #cutoff is in nm
        super().__init__(dna, protein)

    def reset(self):
        exclusionForce = openmm.CustomNonbondedForce(f"""k_exclusion_protein_DNA*energy;
                         energy=(4*epsilon*((sigma/r)^12-(sigma/r)^6)-offset)*step(cutoff-r);
                         offset=4*epsilon*((sigma/cutoff)^12-(sigma/cutoff)^6);
                         sigma=0.5*(sigma1+sigma2); 
                         epsilon=sqrt(epsilon1*epsilon2);
                         cutoff=sqrt(cutoff1*cutoff2)""")
        exclusionForce.addGlobalParameter('k_exclusion_protein_DNA', self.k)
        exclusionForce.addPerParticleParameter('epsilon')
        exclusionForce.addPerParticleParameter('sigma')
        exclusionForce.addPerParticleParameter('cutoff')
        exclusionForce.setCutoffDistance(self.cutoff)
        # exclusionForce.setUseLongRangeCorrection(True)
        exclusionForce.setForceGroup(self.force_group)  # There can not be multiple cutoff distance on the same force group
        if self.periodic:
            exclusionForce.setNonbondedMethod(exclusionForce.CutoffPeriodic)
        else:
            exclusionForce.setNonbondedMethod(exclusionForce.CutoffNonPeriodic)
        print(f"protein dna cutoff {exclusionForce.getCutoffDistance()}")
        self.force = exclusionForce

    def defineInteraction(self):

        particle_definition = self.dna.config['Protein-DNA particles']
        dna_particle_definition=particle_definition[(particle_definition['molecule'] == 'DNA') &
                                                    (particle_definition['DNA'] == self.dna.DNAtype)]
        protein_particle_definition = particle_definition[(particle_definition['molecule'] == 'Protein')]

        # Merge DNA and protein particle definitions
        particle_definition = pandas.concat([dna_particle_definition, protein_particle_definition], sort=False)
        particle_definition.index = particle_definition.molecule + particle_definition.name
        self.particle_definition = particle_definition

        is_dna = self.dna.atoms['resname'].isin(_dnaResidues)
        is_protein = self.dna.atoms['resname'].isin(_proteinResidues)
        atoms = self.dna.atoms.copy()
        atoms['is_dna'] = is_dna
        atoms['is_protein'] = is_protein
        atoms['epsilon']=np.nan
        atoms['radius']=np.nan
        atoms['cutoff'] = np.nan
        DNA_list = []
        protein_list = []
        for i, atom in atoms.iterrows():
            if atom.is_dna:
                param = particle_definition.loc['DNA' + atom['name']]
                if self.radius_override == None:
                    parameters = [param.epsilon,
                                param.radius,
                                param.cutoff]
                else:
                    parameters = [param.epsilon,
                                self.radius_override,
                                param.cutoff]
                DNA_list += [i]
                #print(i,parameters)
            elif atom.is_protein:
                param = particle_definition.loc['Protein' + atom['name']]
                if self.radius_override == None:
                    parameters = [param.epsilon,
                                param.radius,
                                param.cutoff]
                else:
                    parameters = [param.epsilon,
                                self.radius_override,
                                param.cutoff]
                protein_list += [i]
                #print(i,parameters)
            else:
                print(f'Residue {i} not included in protein-DNA interactions')
                parameters = [0, .1,.1]
            atoms.loc[i, ['epsilon', 'radius', 'cutoff']] = parameters
            self.atoms = atoms
            self.force.addParticle(parameters)
        self.force.addInteractionGroup(DNA_list, protein_list)

        # addExclusions
        addNonBondedExclusions(self.dna, self.force)


class ElectrostaticsProteinDNA(ProteinDNAForce):
    """DNA-protein and protein-protein electrostatics."""
    def __init__(self, dna, protein, k=1, ldby = 1.2 * unit.nanometer, cutoff_distance = None, force_group=15):
        self.k = k
        self.force_group = force_group
        self.ldby = ldby
        self.cutoff_distance = cutoff_distance
        super().__init__(dna, protein)

    def reset(self):
        dielectric = 78 # e * a
        #print(dielectric)
        # Debye length
        Na = unit.AVOGADRO_CONSTANT_NA  # Avogadro number
        ec = 1.60217653E-19 * unit.coulomb  # proton charge
        pv = 8.8541878176E-12 * unit.farad / unit.meter  # dielectric permittivity of vacuum

        #ldby = 1.2 * unit.nanometer # np.sqrt(dielectric * pv * kb * T / (2.0 * Na * ec ** 2 * C))
        denominator = 4 * np.pi * pv * dielectric / (Na * ec ** 2)
        denominator = denominator.in_units_of(unit.kilocalorie_per_mole**-1 * unit.nanometer**-1)
        #print(ldby, denominator)
        k = self.k
        ldby = self.ldby
        electrostaticForce = openmm.CustomNonbondedForce(f"""k_electro_protein_DNA*energy;
                             energy=q1*q2*exp(-r/inter_dh_length)/inter_denominator/r;""")
        electrostaticForce.addPerParticleParameter('q')
        electrostaticForce.addGlobalParameter('k_electro_protein_DNA', k)
        electrostaticForce.addGlobalParameter('inter_dh_length', ldby)
        electrostaticForce.addGlobalParameter('inter_denominator', denominator)

        if self.cutoff_distance == None:
            cutoff_distance = ldby * 4
        else:
            cutoff_distance = self.cutoff_distance

        cutoff_nm = cutoff_distance.value_in_unit(unit.nanometer)

        electrostaticForce.setCutoffDistance(cutoff_nm)
        print(f"protein dna screening length {ldby} nm")
        print(f"protein dna cutoff {electrostaticForce.getCutoffDistance()} nm")
        if self.periodic:
            electrostaticForce.setNonbondedMethod(electrostaticForce.CutoffPeriodic)
        else:
            electrostaticForce.setNonbondedMethod(electrostaticForce.CutoffNonPeriodic)
        electrostaticForce.setForceGroup(self.force_group)
        self.force = electrostaticForce

    def defineInteraction(self):
        # Merge DNA and protein particle definitions
        particle_definition = self.dna.config['Protein-DNA particles']
        dna_particle_definition=particle_definition[(particle_definition['molecule'] == 'DNA') &
                                                    (particle_definition['DNA'] == self.dna.DNAtype)]
        protein_particle_definition = particle_definition[(particle_definition['molecule'] == 'Protein')]

        # Merge DNA and protein particle definitions
        particle_definition = pandas.concat([dna_particle_definition, protein_particle_definition], sort=False)
        particle_definition.index = particle_definition.molecule + particle_definition.name
        self.particle_definition = particle_definition

        # Open Sequence dependent electrostatics
        sequence_electrostatics = self.dna.config['Sequence dependent electrostatics']
        sequence_electrostatics.index = sequence_electrostatics.resname

        # Select only dna and protein atoms
        is_dna = self.protein.atoms['resname'].isin(_dnaResidues)
        is_protein = self.protein.atoms['resname'].isin(_proteinResidues)
        atoms = self.protein.atoms.copy()
        atoms['is_dna'] = is_dna
        atoms['is_protein'] = is_protein
        DNA_list = []
        protein_list = []

        for i, atom in atoms.iterrows():
            if atom.is_dna:
                param = particle_definition.loc['DNA' + atom['name']]
                charge = param.charge
                parameters = [charge]
                if charge != 0:
                    DNA_list += [i]
                    #print(atom.chainID, atom.resSeq, atom.resname, atom['name'], charge)
            elif atom.is_protein:
                atom_param = particle_definition.loc['Protein' + atom['name']]
                seq_param = sequence_electrostatics.loc[atom.real_resname]
                charge = atom_param.charge * seq_param.charge
                parameters = [charge]
                if charge != 0:
                    protein_list += [i]
                    #print(atom.chainID, atom.resSeq, atom.resname, atom['name'], charge)
            else:
                print(f'Residue {i} not included in protein-DNA electrostatics')
                parameters = [0]  # No charge if it is not DNA
            # print (i,parameters)
            self.force.addParticle(parameters)
        self.force.addInteractionGroup(DNA_list, protein_list)
        # self.force.addInteractionGroup(protein_list, protein_list) #protein-protein electrostatics should be included using debye Huckel Terms

        # addExclusions
        addNonBondedExclusions(self.dna, self.force)

class BiasElectrostaticsProteinDNA(ProteinDNAForce):
    """ Protein-DNA string potential"""
    #k_ebias and center should be inputted
    def __init__(self, dna, protein, k_ebias,center, k_elec, ldby, cutoff_distance = None, forceGroup=16):
        self.k_ebias = k_ebias
        self.center = center
        self.k_elec = k_elec
        self.ldby = ldby
        self.forceGroup = forceGroup
        self.cutoff_distance = cutoff_distance
        super().__init__(dna, protein)

    def reset(self):
        #k_ebias=self.k_ebias.value_in_unit(unit.kilojoule_per_mole)
        #center=self.center.value_in_unit(unit.kilojoule_per_mole)
        k_ebias = self.k_ebias
        center = self.center
        ebiasForce = openmm.CustomCVForce(f"0.5*k_ebias*((E_elec-center)/4.184)^2")
        E_elec = ElectrostaticsProteinDNA(self.dna, self.protein, k = self.k_elec, ldby = self.ldby, cutoff_distance = self.cutoff_distance)
        elec = E_elec.force
        #ebiasForce.addCollectiveVariable("E_elec", E_elec)
        ebiasForce.addCollectiveVariable("E_elec", elec)    #Is in kJ/mol
        ebiasForce.addGlobalParameter("k_ebias", k_ebias)   #
        ebiasForce.addGlobalParameter("center", center)
        ebiasForce.setForceGroup(self.forceGroup)
        print(f"k_ebias = {k_ebias}")
        print(f"center = {center}")
        #print (E_elec)
        self.force = ebiasForce

    def defineInteraction(self):
        print(f"ElectrostaticsProteinDNA bias on: center at {self.center}, k_ebias = {self.k_ebias}, with electrostatic parameters k_elec = {self.k_elec} and screening length {self.ldby}")

class AMHgoProteinDNA(ProteinDNAForce):
    """ Protein-DNA amhgo potential"""
    def __init__(self, dna, protein, chain_protein='A', chain_DNA='B', k_amhgo_PD=1*unit.kilocalorie_per_mole, sigma_sq=0.05*unit.nanometers**2, aaweight=False, globalct=True, cutoff=1.8, force_group=16):
        self.force_group = force_group
        self.k_amhgo_PD = k_amhgo_PD
        self.sigma_sq= sigma_sq
        self.chain_protein = chain_protein
        self.chain_DNA = chain_DNA
        self.aaweight = aaweight
        self.cutoff = cutoff
        self.globalct = globalct
        super().__init__(dna, protein)

    def reset(self):
        cutoff = self.cutoff
        k_3spn2 = self.k_3spn2
        if self.globalct:
                amhgoForce = openmm.CustomBondForce(f"-k_amhgo_PD*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_sq))*step({cutoff}-r)")
        else:
                amhgoForce = openmm.CustomBondForce(f"-k_amhgo_PD*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_sq))*step(r_ijN+{cutoff}-r)")
        amhgoForce.addGlobalParameter("k_amhgo_PD", k_3spn2*self.k_amhgo_PD)
        amhgoForce.addGlobalParameter("sigma_sq", self.sigma_sq)
        amhgoForce.addPerBondParameter("gamma_ij")
        amhgoForce.addPerBondParameter("r_ijN")
        amhgoForce.setUsesPeriodicBoundaryConditions(self.periodic)
        amhgoForce.setForceGroup(self.force_group)  # There can not be multiple cutoff distance on the same force group
        self.force = amhgoForce

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index
        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'])

        contact_list = np.loadtxt("contact_protein_DNA.dat")
        for i in range(len(contact_list)):
            if self.aaweight:
                gamma_ij = contact_list[i][3]
            else:
                gamma_ij = 1.0
            if (self.chain_protein, int(contact_list[i][0]), 'CB') in atoms.index:
                 CB_protein = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['resSeq'] == int(contact_list[i][0])) & (atoms['name'] == 'CB') & atoms['resname'].isin(_proteinResidues)].copy()
            else:
                 CB_protein = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['resSeq'] == int(contact_list[i][0])) & (atoms['name'] == 'CA') & atoms['resname'].isin(_proteinResidues)].copy()
            base_DNA = atoms[(atoms['chainID'] == self.chain_DNA) & (atoms['resSeq'] == int(contact_list[i][1])) & (atoms['name'].isin(['A', 'T', 'G', 'C'])) & atoms['resname'].isin(_dnaResidues)].copy()
            r_ijN = contact_list[i][2]/10.0*unit.nanometers
            self.force.addBond(int(CB_protein['index'].values[0]), int(base_DNA['index'].values[0]), [gamma_ij, r_ijN])
            print(int(CB_protein['index'].values[0]), int(base_DNA['index'].values[0]), [gamma_ij, r_ijN])

class NFkBBasePairBias(ProteinDNAForce):
    """constrains protein to a particular base pair and its nearest neighbors"""
    def __init__(self, dna, protein, indices, forceGroup=16):
        self.forceGroup = forceGroup
        assert len(indices)==12, indices
        self.indices = indices
        super().__init__(dna,protein)
    def reset(self):
        """
        # got unhelpful segfault with this function as originally written so 
        # experimenting here
        self.force = openmm.CustomCompoundBondForce(10,'1')
        print(f'self.indices: {self.indices}')
        self.force.addBond(list(range(10))
        """
        # mathematically, E1 and E2 are even (E1(x)==E1(-x) and E2(x)==E2(-x))
        E1 = "(4.184*(5*(tanh(30*((x)-(1/2)))+tanh(30*(-(x)-(1/2))))+10))" # shifted so that minimum is y=0
        E2 = "(4.184*(50*(tanh(30*((x)-(1/2)))+tanh(30*(-(x)-(1/2))))+100))" # shifted so that minimum is y=0
        # but negative arguments don't make sense because we only want these to activate
        # when the component of the (protein-bp1) vector along the (bp2-bp1) vector is positive,
        # so we multiply by the openmm step() function, which is 1 when x>=0 and 0 otherwise.
        E1_positive = f'step(x)*{E1}'
        E2_positive = f'step(x)*{E2}'
        #     this does not lead to any differentiability issues because E1, E2, and their derivatives are 0 at x=0.
        # Now, we sum the energy contributions from the (protein-im1)->(im2-im1), (protein-i)->(im1-i), (protein-i)->(ip1-i), and (protein-ip1)->(ip2-ip1) projections,
        #     where im1 is the coordinates of base pair i-1, ip1 is the coordinates of base pair i+1, etc.
        # If the protein is between base pairs i-1 and i+1, we expect one of the four components to be positive (either ip1 or im1)
        #     and therefore have its energy considered (have its step function activated).
        #     The energy will be about 0 if the target base pair, i, is the closest base pair, while there will be a penalty 
        #     if i-1 or i+1 is closer than i.
        # If the protein is between i-1 and i-2 or i+1 and i+2, then we expect both i+1 and i+2 to be activated (meaning their step functions equal 1),
        #    or both i-1 and i-2 could be activated. The protein will always pay the E1 penalty. It will pay the E2 penalty when it is
        #    closer to i-2 than i-1 (or closer to i+2 than i+1).
        ####################################################################################################################################
        energy = f'{E1.replace("x","theta_0inf_comp_ip1")}+{E1.replace("x","theta_0inf_comp_im1")}+{E2.replace("x","theta_0inf_comp_ip2")}+{E2.replace("x","theta_0inf_comp_ip2")}'
        #energy = '4.184*pointdistance(bx,by,bz,proteinx,proteiny,proteinz)'
        ########################################################################################################################################
        # define switching function that turns on (quickly goes from 0 to 1) when input is between 0 and infinity
        theta_0inf = '(1/2)*(tanh(70*x)+1)'
        # plug components of DNA-protein vectors along DNA-DNA vectors into theta
        theta_definitions = f';theta_0inf_comp_ip1={theta_0inf.replace("x","comp_ip1")};theta_0inf_comp_im1={theta_0inf.replace("x","comp_im1")};theta_0inf_comp_ip2={theta_0inf.replace("x","comp_ip2")};theta_0inf_comp_im2={theta_0inf.replace("x","comp_im2")}'
        # define components based on dot product
        # p1, p2: phosphates on i-2
        # p3, p4: phosphates on i-1
        # p5, p6: phosphates on i
        # p7, p8: phosphates on i+1
        # p9, p10: phosphates on i+2
        # p11, p12: DD residues on opposites sides of interface
        #comp_definitions = ';comp_im2=im2im1x*proteinbm1x+im2im1y*proteinbm1y+im2im1z*proteinbm1z'
        #                # im2 vector - im1 vector                             im1 vector - i vector                     # ip1 vector - i vector                   # ip2 vector - ip1 vector
        #differences = ';im2im1x=bm2x-bm1x;im2im1y=bm2y-bm1y;im2im1z=bm2z-bm1z;im1ix=bm1x-bx;im1iy=bm1y-by;im1iz=bm1z-bz;ip1ix=bp1x-bx;ip1iy=bp1y-by;ip1iz=bp1z-bz;ip2ip1x=bp2x-bp1x;ip2ip1y=bp2y-bp1y;ip2ip1z=bp2z-bp1z;proteinbm1x=proteinx-bm1x;proteinbm1y=proteiny-bm1y;proteinbm1z=proteinz-bm1z;proteinbx=proteinx-bx;proteinby=proteiny-by;proteinbz=proteinz-bz;proteinbp1x=proteinx-bp1x;proteinbp1y=proteiny-bp1y;proteinbp1z=proteinz-bp1z'
        #avg_definitions = ';bm2x=(x1+x2)/2;bm2y=(y1+y2)/2;bm2z=(z1+z2)/2;bm1x=(x3+x4)/2;bm1y=(y3+y4)/2;bm1z=(z3+z4)/2;bx=(x5+x6)/2;by=(y5+y6)/2;bz=(z5+z6)/2;bp1x=(x7+x8)/2;bp1y=(y7+y8)/2;bp1z=(z7+z8)/2;bp2x=(x9+10)/2;bp2y=(y9+y10)/2;bp2z=(z9+z10)/2;proteinx=(x11+x12)/2;proteiny=(y11+y12)/2;proteinz=(z11+z12)/2'
        #force = openmm.CustomCompoundBondForce(12,f'{energy}{theta_definitions}{comp_definitions}{differences}{avg_definitions}')

        comp_definitions=';comp_ip1=pointdistance(bx,by,bz,proteinx,proteiny,proteinz)*cos(pointangle(proteinx,proteiny,proteinz,bx,by,bz,bp1x,bp1y,bp1z))/pointdistance(bx,by,bz,bp1x,bp1y,bp1z)\
;comp_im1=pointdistance(bx,by,bz,proteinx,proteiny,proteinz)*cos(pointangle(proteinx,proteiny,proteinz,bx,by,bz,bm1x,bm1y,bm1z))/pointdistance(bx,by,bz,bm1x,bm1y,bm1z)\
;comp_ip2=pointdistance(bp1x,bp1y,bp1z,proteinx,proteiny,proteinz)*cos(pointangle(proteinx,proteiny,proteinz,bp1x,bp1y,bp1z,bp2x,bp2y,bp2z))/pointdistance(bp1x,bp1y,bp1z,bp2x,bp2y,bp2z)\
;comp_im2=pointdistance(bm1x,bm1y,bm1z,proteinx,proteiny,proteinz)*cos(pointangle(proteinx,proteiny,proteinz,bm1x,bm1y,bm1z,bm2x,bm2y,bm2z))/pointdistance(bm1x,bm1y,bm1z,bm2x,bm2y,bm2z)'
        avg_definitions = ';bm2x=(x1+x2)/2;bm2y=(y1+y2)/2;bm2z=(z1+z2)/2;bm1x=(x3+x4)/2;bm1y=(y3+y4)/2;bm1z=(z3+z4)/2;bx=(x5+x6)/2;by=(y5+y6)/2;bz=(z5+z6)/2;bp1x=(x7+x8)/2;bp1y=(y7+y8)/2;bp1z=(z7+z8)/2;bp2x=(x9+10)/2;bp2y=(y9+y10)/2;bp2z=(z9+z10)/2;proteinx=(x11+x12)/2;proteiny=(y11+y12)/2;proteinz=(z11+z12)/2'
        force = openmm.CustomCompoundBondForce(12,f'{energy}{theta_definitions}{comp_definitions}{avg_definitions}')
        force.addBond(self.indices)
        #force.setUsesPeriodicBoundaryConditions(True)
        force.setForceGroup(16)
        self.force = force
         
    def defineInteraction(self):
        pass

#class AMHgoProteinDNA(ProteinDNAForce):
#    """ Protein-DNA amhgo potential (Xinyu)"""
#    def __init__(self, dna, protein, chain_protein='A', chain_DNA='B', k_amhgo_PD=1*unit.kilocalorie_per_mole,
#                 sigma_sq=0.05*unit.nanometers**2, aaweight=False, cutoff=1.8, force_group=16):
#        self.force_group = force_group
#        self.k_amhgo_PD = k_amhgo_PD
#        self.sigma_sq= sigma_sq
#        self.chain_protein = chain_protein
#        self.chain_DNA = chain_DNA
#        self.aaweight = aaweight
#        self.cutoff = cutoff
#        super().__init__(dna, protein)
#
#    def reset(self):
#        cutoff = self.cutoff
#        amhgoForce = openmm.CustomBondForce(f"-k_amhgo_PD*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_sq))*step({cutoff}-r)")
#        amhgoForce.addGlobalParameter("k_amhgo_PD", self.k_amhgo_PD)
#        amhgoForce.addGlobalParameter("sigma_sq", self.sigma_sq)
#        amhgoForce.addPerBondParameter("gamma_ij")
#        amhgoForce.addPerBondParameter("r_ijN")
#        amhgoForce.setUsesPeriodicBoundaryConditions(self.periodic)
#        amhgoForce.setForceGroup(self.force_group)  # There can not be multiple cutoff distance on the same force group
#        self.force = amhgoForce
#
#    def defineInteraction(self):
#        atoms = self.dna.atoms.copy()
#        atoms['index'] = atoms.index
#        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'])
#
#        contact_list = np.loadtxt("contact_protein_DNA.dat")
#        for i in range(len(contact_list)):
#            if self.aaweight: gamma_ij = contact_list[i][3]
#            else:   gamma_ij = 1.0
#            if (self.chain_protein, int(contact_list[i][0]), 'CB') in atoms.index:
#                 CB_protein = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['resSeq'] == int(contact_list[i][0])) &
#                                    (atoms['name'] == 'CB') & atoms['resname'].isin(_proteinResidues)].copy()
#            else:
#                 CB_protein = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['resSeq'] == int(contact_list[i][0])) &
#                                    (atoms['name'] == 'CA') & atoms['resname'].isin(_proteinResidues)].copy()
#            base_DNA = atoms[(atoms['chainID'] == self.chain_DNA) & (atoms['resSeq'] == int(contact_list[i][1])) & (atoms['name'].isin(['A', 'T', 'G', 'C'])) & atoms['resname'].isin(_dnaResidues)].copy()
#            r_ijN = contact_list[i][2]/10.0*unit.nanometers
#            self.force.addBond(int(CB_protein['index'].values[0]), int(base_DNA['index'].values[0]), [gamma_ij, r_ijN])
#            print(int(CB_protein['index'].values[0]), int(base_DNA['index'].values[0]), [gamma_ij, r_ijN])

'''  Steven Luo ordering that Xinyu's protein-DNA version be deprecated but retained.
class StringProteinDNA(ProteinDNAForce):
    """ Protein-DNA string potential (Xinyu)"""
    def __init__(self, dna, protein, r0, chain_protein='A', chain_DNA='B', k_string_PD=10*4.184, protein_seg=False, group=[]):
        self.k_string_PD = k_string_PD
        self.chain_protein = chain_protein
        self.chain_DNA = chain_DNA
        self.r0 = r0
        self.protein_seg = protein_seg
        self.group = group
        super().__init__(dna, protein)

    def reset(self):
        r0=self.r0
        k_string_PD=self.k_string_PD
        stringForce = openmm.CustomCentroidBondForce(2, f"0.5*{k_string_PD}*(distance(g1,g2)-{r0})^2")
        self.force = stringForce
        print("String_PD bias on: r0, k_string = ", r0, k_string_PD)

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index
        CA_atoms = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['name'] == 'CA') & atoms['resname'].isin(_proteinResidues)].copy()
        S_atoms = atoms[(atoms['chainID'] == self.chain_DNA) & (atoms['name'] == 'S') & atoms['resname'].isin(_dnaResidues)].copy()
        CA_index = [int(atom.index) for atom in CA_atoms.itertuples()]
        if self.protein_seg: self.force.addGroup([CA_index[x] for x in self.group])
        else:   self.force.addGroup(CA_index)
        self.force.addGroup([int(atom.index) for atom in S_atoms.itertuples()])
        bondGroups = [0, 1]
        print(self.force.getGroupParameters(0))
        print(self.force.getGroupParameters(1))

        self.force.addBond(bondGroups)
'''

class MultiChainProteinDNA(ProteinDNAForce):
    """ Protein-DNA string potential for multiple chains (added and amended by Steven and ChatGPT from Xinyu)"""
    def __init__(self, dna, protein, r0, chain_protein='AB', chain_DNA='CD', k_string_PD=10*4.184, protein_seg=False, group=[], force_group=19): #forceGroup placeholder is 19.
        self.k_string_PD = k_string_PD
        self.chain_protein = chain_protein
        self.chain_DNA = chain_DNA
        self.r0 = r0
        self.protein_seg = protein_seg
        self.group = group
        self.force_group = force_group
        super().__init__(dna, protein)

    def reset(self):
        r0=self.r0
        k_string_PD=self.k_string_PD
        stringForce = openmm.CustomCentroidBondForce(2, f"0.5*{k_string_PD}*(distance(g1,g2)-{r0})^2")
        stringForce.setForceGroup(self.force_group)
        self.force = stringForce
        print("String_PD bias on: r0, k_string = ", r0, k_string_PD)

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index

        CA_atoms = atoms[
            atoms['chainID'].isin(list(self.chain_protein)) &
            (atoms['name'] == 'CA') &
            atoms['resname'].isin(_proteinResidues)
        ].copy()

        S_atoms = atoms[
            atoms['chainID'].isin(list(self.chain_DNA)) &
            (atoms['name'] == 'S') &
            atoms['resname'].isin(_dnaResidues)
        ].copy()

        if CA_atoms.empty or S_atoms.empty:
            raise ValueError("No CA or S atoms found in the specified chains.")

        CA_index = [int(atom.index) for atom in CA_atoms.itertuples()]
        if self.protein_seg:
            self.force.addGroup([CA_index[x] for x in self.group])
        else:
            self.force.addGroup(CA_index)

        self.force.addGroup([int(atom.index) for atom in S_atoms.itertuples()])
        self.force.addBond([0, 1])

        print("Protein group:", self.force.getGroupParameters(0))
        print("DNA group:", self.force.getGroupParameters(1))

class String_length_ProteinDNA(ProteinDNAForce):
    """ Protein-DNA string potential (Xinyu)"""
    def __init__(self, dna, protein, chain_protein='A', chain_DNA='B', protein_seg=False, group=[], force_group=17):
        self.force_group = force_group
        self.chain_protein = chain_protein
        self.chain_DNA = chain_DNA
        self.protein_seg = protein_seg
        self.group = group
        super().__init__(dna, protein)

    def reset(self):
        length = openmm.CustomCentroidBondForce(2, "distance(g1,g2)")
        length.setForceGroup(self.force_group)
        self.force = length

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index
        CA_atoms = atoms[(atoms['chainID'] == self.chain_protein) & (atoms['name'] == 'CA') & atoms['resname'].isin(_proteinResidues)].copy()
        S_atoms = atoms[(atoms['chainID'] == self.chain_DNA) & (atoms['name'] == 'S') & atoms['resname'].isin(_dnaResidues)].copy()
        CA_index = [int(atom.index) for atom in CA_atoms.itertuples()]
        if self.protein_seg: self.force.addGroup([CA_index[x] for x in self.group])
        else:   self.force.addGroup(CA_index)
        self.force.addGroup([int(atom.index) for atom in S_atoms.itertuples()])
        bondGroups = [0, 1]
        print(self.force.getGroupParameters(0))
        print(self.force.getGroupParameters(1))

        self.force.addBond(bondGroups)

class Multi_length_ProteinDNA(ProteinDNAForce):
    """ Protein-DNA string length for multiple chains (added and amended by Steven and ChatGPT from Xinyu)"""
    def __init__(self, dna, protein, chain_protein='AB', chain_DNA='CD', protein_seg=False, group=[], force_group=4): #force group placeholder is 4
        self.force_group = force_group
        self.chain_protein = chain_protein
        self.chain_DNA = chain_DNA
        self.protein_seg = protein_seg
        self.group = group
        super().__init__(dna, protein)

    def reset(self):
        length = openmm.CustomCentroidBondForce(2, "distance(g1,g2)")
        length.setForceGroup(self.force_group)
        self.force = length

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index

        CA_atoms = atoms[
            atoms['chainID'].isin(list(self.chain_protein)) &
            (atoms['name'] == 'CA') &
            atoms['resname'].isin(_proteinResidues)
        ].copy()

        S_atoms = atoms[
            atoms['chainID'].isin(list(self.chain_DNA)) &
            (atoms['name'] == 'S') &
            atoms['resname'].isin(_dnaResidues)
        ].copy()

        if CA_atoms.empty or S_atoms.empty:
            raise ValueError("No CA or S atoms found in the specified chains.")

        CA_index = [int(atom.index) for atom in CA_atoms.itertuples()]
        if self.protein_seg:
            self.force.addGroup([CA_index[x] for x in self.group])
        else:
            self.force.addGroup(CA_index)

        self.force.addGroup([int(atom.index) for atom in S_atoms.itertuples()])
        self.force.addBond([0, 1])

        print("Protein group:", self.force.getGroupParameters(0))
        print("DNA group:", self.force.getGroupParameters(1))
