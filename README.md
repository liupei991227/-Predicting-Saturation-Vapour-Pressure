## About the data
The term project is based on the GeckoQ dataset with atomic structures of 31,637 atmospherically relevant molecules resulting from the oxidation of 𝛼-pinene, toluene and decane. The GeckoQ dataset is built to
complement data-driven research in atmospheric science. It provides molecular data relevant to aerosol particle growth and new particle formation. A key molecular property related to aerosol particle growth is the
saturation vapour pressure (pSat), a measure of a molecule’s ability to condense to the liquid phase. Molecules with low pSat, low-volatile organic compounds (LVOC) are particularly interesting for NPF research. All the
data in GeckoQ pertains to LVOCs. (For more information, see: Besel et al. https://doi.org/10.1038/s41597-023-02366-x).

For each molecule, GeckoQ features important thermodynamic properties: saturation vapour pressure (pSat), the chemical potential [kJ/mol], the free energy of a molecule in mixture [kJ/mol], and the heat of vaporisation
[kJ/mol]. Out of these, the logarithmic saturation vapour pressure will be the focus of your term project. The logarithmic scale is used instead of raw pSat to bring the quantities to more manageable range. There
are two types of features that you will have the choice of using for your project: the interpretable features (described in detail below) and the topographical footprints (TopFP) of the molecules. Previous works record
using the TopFP descriptor as inputs to a machine learning model to learn pSat as a function of atomic structure for a different dataset. (Wang et al. https://doi.org/10.1073/pnas.1707564114)
Following are the columns which make up the training/test data set. Barring the Id and the log_pSat_Pa columns, all others form the interpretable features of the molecules:

• ID - A unique molecule index used in naming files.

• log_pSat_Pa- Logarithmic saturation vapour pressure of the molecule calculated by COSMOtherm (Pa).

• MW - The molecular weight of the molecule (g/mol).

• NumOfAtoms - The number of atoms in the molecule.

• NumOfC - The number of carbon atoms in the molecule.

• NumOfO- The number of oxygen atoms in the molecule.

• NumOfN- The number of nitrogen atoms in the molecule.

• NumHBondDonors - “The number of hydrogen bond donors in the molecule, i.e. hydrogens bound to oxygen.”

• parentspecies- Either “decane”, “toluene”, “apin” for alpha-pinene or a combination of these connected by an underscore to indicate ambiguous descent. In 243 cases, the parent species is “None” because it was not possible to retrieve it.

• NumOfConf- The number of stable conformers found and successfully calculated by COSMOconf.

• NumOfConfUsed- The number of conformers used to calculate the thermodynamic properties.

• C = C (non-aromatic)- The number of non-aromatic C=C bounds found in the molecule.

• C = C-C = O in non-aromatic ring- The number of “C=C-C=O” structures found in non-aromatic rings in the molecule.

• hydroxyl (alkyl) - The number of the alkylic hydroxyl groups found in the molecule.

• aldehyde- The number of aldehyde groups in the molecule.

• ketone - The number of ketone groups in the molecule.

• carboxylic acid - The number of carboxylic acid groups in the molecule.

• ester - The number of ester groups in the molecule.

• ether (alicyclic)- The number of alicyclic ester groups in the molecule.

• nitrate - The number of alicyclic nitrate groups in the molecule.

• nitro - The number of nitro ester groups in the molecule.

• aromatic hydroxyl - The number of alicyclic aromatic hydroxyl groups in the molecule.

• carbonylperoxynitrate - The number of carbonylperoxynitrate groups in the molecule.

• peroxide - The number of peroxide groups in the molecule.

• hydroperoxide- The number of hydroperoxide groups in the molecule.

• carbonylperoxyacid- The number of carbonylperoxyacid groups found in the molecule

• nitroester- The number of nitroester groups found in the molecule

## Your task
Saturation vapour pressure is a continuous variable, hence, your task is to build a regression-based machine learning model that uses the aforementioned interpretable features or topographical fingerprints of the molecules.

NOTE: This is a non-trivial regression task. It is possible to do it in many ways. The most straightforward regression model that you could build is a linear regressor, but that is inefficient for the task because the relationship between input features and the pSat is non-linear. Therefore, you should undertake a thorough data exploration, pre processing, feature selection, model selection, performance estimation, etc., appropriately since you will report and analyse your choices and results in the term project report.

The project’s purpose is not to (even try to!) replicate any methods in the literature, make a super-complex best-performing classifier that beats everything else or attempt to use other data sources, etc., to obtain the best possible performance score. You should not use any method that you do not understand yourself!

Accuracy of the predictions on the test data is not a grading criterion by itself, even though a terrible performance may indicate something fishy in your approach (which can affect grading).






