"""Class counts of the UCR univariate archive, from its ``DataSummary.csv``.

Names follow tslearn, which spells NonInvasiveFatalECGThorax1/2.
Solvers read them through ``Dataset.meta`` to skip a dataset without loading it.
Source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv
"""

# fmt: off
UCR_N_CLASSES = {
    "ACSF1": 10, "Adiac": 37, "AllGestureWiimoteX": 10, "AllGestureWiimoteY": 10,
    "AllGestureWiimoteZ": 10, "ArrowHead": 3, "BME": 3, "Beef": 5, "BeetleFly": 2,
    "BirdChicken": 2, "CBF": 3, "Car": 4, "Chinatown": 2, "ChlorineConcentration": 3,
    "CinCECGTorso": 4, "Coffee": 2, "Computers": 2, "CricketX": 12, "CricketY": 12,
    "CricketZ": 12, "Crop": 24, "DiatomSizeReduction": 4,
    "DistalPhalanxOutlineAgeGroup": 3, "DistalPhalanxOutlineCorrect": 2,
    "DistalPhalanxTW": 6, "DodgerLoopDay": 7, "DodgerLoopGame": 2,
    "DodgerLoopWeekend": 2, "ECG200": 2, "ECG5000": 5, "ECGFiveDays": 2,
    "EOGHorizontalSignal": 12, "EOGVerticalSignal": 12, "Earthquakes": 2,
    "ElectricDevices": 7, "EthanolLevel": 4, "FaceAll": 14, "FaceFour": 4,
    "FacesUCR": 14, "FiftyWords": 50, "Fish": 7, "FordA": 2, "FordB": 2,
    "FreezerRegularTrain": 2, "FreezerSmallTrain": 2, "Fungi": 18,
    "GestureMidAirD1": 26, "GestureMidAirD2": 26, "GestureMidAirD3": 26,
    "GesturePebbleZ1": 6, "GesturePebbleZ2": 6, "GunPoint": 2, "GunPointAgeSpan": 2,
    "GunPointMaleVersusFemale": 2, "GunPointOldVersusYoung": 2, "Ham": 2,
    "HandOutlines": 2, "Haptics": 5, "Herring": 2, "HouseTwenty": 2, "InlineSkate": 7,
    "InsectEPGRegularTrain": 3, "InsectEPGSmallTrain": 3, "InsectWingbeatSound": 11,
    "ItalyPowerDemand": 2, "LargeKitchenAppliances": 3, "Lightning2": 2,
    "Lightning7": 7, "Mallat": 8, "Meat": 3, "MedicalImages": 10,
    "MelbournePedestrian": 10, "MiddlePhalanxOutlineAgeGroup": 3,
    "MiddlePhalanxOutlineCorrect": 2, "MiddlePhalanxTW": 6,
    "MixedShapesRegularTrain": 5, "MixedShapesSmallTrain": 5, "MoteStrain": 2,
    "NonInvasiveFatalECGThorax1": 42, "NonInvasiveFatalECGThorax2": 42, "OSULeaf": 6,
    "OliveOil": 4, "PLAID": 11, "PhalangesOutlinesCorrect": 2, "Phoneme": 39,
    "PickupGestureWiimoteZ": 10, "PigAirwayPressure": 52, "PigArtPressure": 52,
    "PigCVP": 52, "Plane": 7, "PowerCons": 2, "ProximalPhalanxOutlineAgeGroup": 3,
    "ProximalPhalanxOutlineCorrect": 2, "ProximalPhalanxTW": 6,
    "RefrigerationDevices": 3, "Rock": 4, "ScreenType": 3, "SemgHandGenderCh2": 2,
    "SemgHandMovementCh2": 6, "SemgHandSubjectCh2": 5, "ShakeGestureWiimoteZ": 10,
    "ShapeletSim": 2, "ShapesAll": 60, "SmallKitchenAppliances": 3, "SmoothSubspace": 3,
    "SonyAIBORobotSurface1": 2, "SonyAIBORobotSurface2": 2, "StarLightCurves": 3,
    "Strawberry": 2, "SwedishLeaf": 15, "Symbols": 6, "SyntheticControl": 6,
    "ToeSegmentation1": 2, "ToeSegmentation2": 2, "Trace": 4, "TwoLeadECG": 2,
    "TwoPatterns": 4, "UMD": 3, "UWaveGestureLibraryAll": 8, "UWaveGestureLibraryX": 8,
    "UWaveGestureLibraryY": 8, "UWaveGestureLibraryZ": 8, "Wafer": 2, "Wine": 2,
    "WordSynonyms": 25, "Worms": 5, "WormsTwoClass": 2, "Yoga": 2,
}
# fmt: on
