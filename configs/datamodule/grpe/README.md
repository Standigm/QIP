# Dataset_configs
version tagging: v{major}\_{minor}\_{patch}

## Version Tagging rule

### Major version: 
A major version change indicates a significant update or change in the dataset that may not be backward compatible

```sql
v1.0.0 - Initial version of the dataset
v2.0.0 - Major update with significant changes or additions to the dataset. e.g. labeling rule, input features format, task type, ...
```

### Minor version: 
A minor version change indicates minor updates or additions to the dataset while maintaining backward compatibility. 


```sql
v1.0.0 - Initial version of the dataset
v1.1.0 - Minor update with additional data, the number of data samples, Data split(train/val/test)
```

### Patch version: 
A patch version change indicates minor bug fixes or corrections in the dataset while maintaining backward compatibility.

```sql
v1.0.0 - Initial version of the dataset
v1.0.1 - Patch version with bug fixes, data/label correction for a specific samples
```


## Group Tagging Rule
starts with tagging code with Year/Month/Day

Tag code:

```
M: Multitask
P: Pre-Training
```

{Tag Code}{index}-YYYYMM
```text
M0: 
CYP/1A2, CYP/2C9, CYP/2C19, CYP/2D6, CYP/3A4, hERG, lipophilicity, MS/human, MS/mouse, permeability, solubility/pbs, nablaDFT

M1:
CYP/1A2, CYP/2C9, CYP/2C19, CYP/2D6, CYP/3A4, hERG, lipophilicity, MS/human, MS/mouse, permeability, solubility/pbs

M2:
hERG, lipophilicity, MS/human, MS/mouse, permeability, solubility/pbs

M3:
CYP/1A2, CYP/2C9, CYP/2C19, CYP/2D6, CYP/3A4, 
DILI, HepaTox, hERG, lipophilicity, LiverTox, 
MS/human, MS/mouse, permeability, 
pKa/acidic, pKa/basic, 
solubility/dmso, solubility/pbs, solubility/water, 
tox21/nr_ahr, tox21/nr_ar, tox21/nr_ar_lbd, tox21/nr_aromatase, tox21/nr_er, tox21/nr_er_lbd, tox21/nr_ppar_gamma, tox21/sr_are, tox21/sr_atad5, tox21/sr_hse, tox21/sr_mmp, tox21/sr_p53, 
PubChemQC, nablaDFT

P0:
nablaDFT

P1:
PubChemQC

P2:
nablaDFT, PubChemQC

```

## Config Tree
```bash
├── dataset_configs
│   ├── CYP
│   │   ├── 1A2
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── 2C19
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── 2C9
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── 2D6
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── 3A4
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   └── README.md
│   ├── Desc
│   │   ├── latest.yaml -> v1_0_0.yaml
│   │   ├── README.md
│   │   └── v1_0_0.yaml
│   ├── DILI
│   │   ├── latest.yaml -> v2_0_0.yaml
│   │   ├── README.md
│   │   └── v2_0_0.yaml
│   ├── HepaTox
│   │   ├── latest.yaml -> v2_0_0.yaml
│   │   ├── README.md
│   │   └── v2_0_0.yaml
│   ├── hERG
│   │   ├── latest.yaml -> v2_2_0.yaml
│   │   ├── README.md
│   │   ├── v2_0_0.yaml
│   │   ├── v2_1_0.yaml
│   │   └── v2_2_0.yaml
│   ├── lipophilicity
│   │   ├── latest.yaml -> v1_0_0.yaml
│   │   ├── README.md
│   │   └── v1_0_0.yaml
│   ├── LiverTox
│   │   ├── latest.yaml -> v2_0_0.yaml
│   │   ├── README.md
│   │   └── v2_0_0.yaml
│   ├── MS
│   │   ├── human
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── mouse
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   └── README.md
│   ├── nablaDFT
│   │   ├── README.md
│   │   └── v1_0_0.yaml
│   ├── permeability
│   │   ├── latest.yaml -> v2_0_0.yaml
│   │   ├── README.md
│   │   └── v2_0_0.yaml
│   ├── pKa
│   │   ├── acidic
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── basic
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   └── README.md
│   ├── PubChemQC
│   │   ├── latest.yaml -> v1_0_0.yaml
│   │   ├── README.md
│   │   └── v1_0_0.yaml
│   ├── solubility
│   │   ├── dmso
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── pbs
│   │   │   ├── latest.yaml -> v2_0_0.yaml
│   │   │   └── v2_0_0.yaml
│   │   ├── README.md
│   │   └── water
│   │       ├── latest.yaml -> v2_0_0.yaml
│   │       └── v2_0_0.yaml
│   └── Tox21
│       ├── nr_ahr
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_ar
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_ar_lbd
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_aromatase
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_er
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_er_lbd
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── nr_ppar_gamma
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── README.md
│       ├── sr_are
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── sr_atad5
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── sr_hse
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       ├── sr_mmp
│       │   ├── latest.yaml -> v2_0_0.yaml
│       │   └── v2_0_0.yaml
│       └── sr_p53
│           ├── latest.yaml -> v2_0_0.yaml
│           └── v2_0_0.yaml
├── M0-230301.yaml
├── M1-230301.yaml
├── M2-230301.yaml
├── M3-230423.yaml
├── multidata.yaml
├── P0-230301.yaml
├── P1-230301.yaml
├── P2-230301.yaml
├── README.md
└── test.yaml

```

