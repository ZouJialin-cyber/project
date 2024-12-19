from enum import Enum


class TechType(Enum):
    ssDNA = 1
    DAPI = 2
    HE = 3
    IF = 4
    Transcriptomics = 5
    Protein = 6
    Null = 7
    UNKNOWN = 10
    # Metabolome = 7


KIT_VERSIONS = (
    'Stereo-seq T FF V1.2',
    'Stereo-seq T FF V1.3',
    'Stereo-CITE T FF V1.0',
    'Stereo-CITE T FF V1.1',
    'Stereo-seq N FFPE V1.0',
)

bPlaceHolder = False
fPlaceHolder = -999.999
iPlaceHolder = -999
sPlaceHolder = '-'


class ErrorCode(Enum):
    qcFail = 1  # image qc failed, return code 1
