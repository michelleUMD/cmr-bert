valve_tags = [
    "mv_regurg",
    "av_regurg",
    "av_stenosis",
    "tv_regurg",
    "pv_regurg",
    "av_bicuspid",
]

valve_tags_with_severity = [
    "mv_regurg",
    "av_regurg",
    "av_stenosis",
    "tv_regurg",
    "pv_regurg",
]


chamber_tags = [
    "ra_dilation",
    "la_dilation",
    "rv_dysfunc",
    "rv_dilation",
    "lv_dysfunc",
    "lv_dilation",
    "lv_hypertrophy",
    "lv_fibrosis",
    "lv_noncompact",
]

chamber_tags_with_severity = [
    "ra_dilation",
    "la_dilation",
    "rv_dysfunc",
    "rv_dilation",
    "lv_dysfunc",
    "lv_dilation",
    "lv_hypertrophy",
    "aortic_dilation",
]

aorta_tags = [
    "aortic_atherosclerosis",
    "aortic_artheritis",
]

other_tags = [
    "papillary_thicken",
    "pericardial_effus",
    "myo_pericarditis",
    "pleura_effus",
    "hcm",
    "infiltrative_cm",
    "amyloid",
    "sarcoid",
    "ischemic_cm",
    "nonischemic_cm",
    "myocardial_infarct",
    "hypertensive_pul_dis"
]

special_tags = [
    "aortic_dilation",
    "lv_lge_pattern",
]

all_tag_names_ordered_with_location = [
    "no_valve_abnorm",
    "no_ventricular_abnorm",
    "no_aortic_abnorm",
    "aortic_dilation_location_asc",
    "aortic_dilation_location_dsc",
    "aortic_dilation_location_root",
    "lv_lge_pattern_pattern_epicardial",
    "lv_lge_pattern_pattern_mid-myocardial",
    "lv_lge_pattern_pattern_subendocardial",
    "lv_lge_pattern_pattern_transmural",
    "myo_pericarditis_type_constrictive",
    "myo_pericarditis_type_myo",
    "myo_pericarditis_type_peri",
    "infiltrative_cm_oth_type_not_oth"
]


all_tag_names_ordered_with_mention = valve_tags + chamber_tags + aorta_tags + other_tags + special_tags
all_tag_names_ordered_with_mention = [tag_name + "_mention" for tag_name in all_tag_names_ordered_with_mention]

all_tag_names_ordered_with_severity = valve_tags_with_severity + chamber_tags_with_severity
all_tag_names_ordered_with_severity = [tag_name + "_severity" for tag_name in all_tag_names_ordered_with_severity]


var_rename_dict = {
    "no_valve_abnorm": "No Valve Abnormality",
    "no_aortic_abnorm": "No Aortic Abnormality",
    "no_ventricular_abnorm": "No Ventricular Abnormality",
    "av_stenosis_mention": "Aortic Valve Stenosis Mention",
    "av_stenosis_severity": "Aortic Valve Stenosis Severity",
    "av_regurg_mention": "Aortic Valve Regurgitation Mention",
    "av_regurg_severity": "Aortic Valve Regurgitation Severity",
    "av_bicuspid_mention": "Aortic Valve Bicuspid Mention",
    "mv_regurg_mention": "Mitral Valve Regurgitation Mention",
    "mv_regurg_severity": "Mitral Valve Regurgitation Severity",
    "pv_regurg_mention": "Pulmonary Valve Regurgitation Mention",
    "pv_regurg_severity": "Pulmonary Valve Regurgitation Severity",
    "tv_regurg_mention": "Tricuspid Valve Regurgitation Mention",
    "tv_regurg_severity": "Tricuspid Valve Regurgitation Severity",
    "aortic_dilation_mention": "Aortic Dilation Mention",
    "aortic_dilation_severity": "Aortic Dilation Severity",
    "aortic_dilation_location_asc": "Aortic Dilation Location Ascending",
    "aortic_dilation_location_dsc": "Aortic Dilation Location Descending",
    "aortic_dilation_location_root": "Aortic Dilation Location Root",
    "aortic_atherosclerosis_mention": "Aortic Atherosclerosis Mention",
    "aortic_artheritis_mention": "Aortic Arteritis Mention",
    "ra_dilation_mention": "Right Atrium Dilation Mention",
    "ra_dilation_severity": "Right Atrium Dilation Severity",
    "la_dilation_mention": "Left Atrium Dilation Mention",
    "la_dilation_severity": "Left Atrium Dilation Severity",
    "rv_dysfunc_mention": "Right Ventricle Dysfunction Mention",
    "rv_dysfunc_severity": "Right Ventricle Dysfunction Severity",
    "rv_dilation_mention": "Right Ventricle Dilation Mention",
    "rv_dilation_severity": "Right Ventricle Dilation Severity",
    "lv_dysfunc_mention": "Left Ventricle Dysfunction Mention",
    "lv_dysfunc_severity": "Left Ventricle Dysfunction Severity",
    "lv_dilation_mention": "Left Ventricle Dilation Mention",
    "lv_dilation_severity": "Left Ventricle Dilation Severity",
    "lv_hypertrophy_mention": "Left Ventricle Hypertrophy Mention",
    "lv_hypertrophy_severity": "Left Ventricle Hypertrophy Severity",
    "lv_lge_pattern_mention": "Left Ventricle Late Gadolinium Enhancement (LGE) Pattern Mention",
    "lv_lge_pattern_pattern_epicardial": "Left Ventricle LGE Pattern Epicardial",
    "lv_lge_pattern_pattern_mid-myocardial": "Left Ventricle LGE Pattern Mid-myocardial",
    "lv_lge_pattern_pattern_subendocardial": "Left Ventricle LGE Pattern Subendocardial",
    "lv_lge_pattern_pattern_transmural": "Left Ventricle LGE Pattern Transmural",
    "lv_fibrosis_mention": "Left Ventricle Fibrosis Mention",
    "lv_noncompact_mention": "Left Ventricle Noncompaction Mention",
    "papillary_thicken_mention": "Papillary Muscle Thickening Mention",
    "pericardial_effus_mention": "Pericardial Effusion Mention",
    "myo_pericarditis_mention": "Myopericarditis Mention",
    "myo_pericarditis_type_constrictive": "Myopericarditis Type Constrictive",
    "myo_pericarditis_type_myo": "Myopericarditis Type Myocarditis",
    "myo_pericarditis_type_peri": "Myopericarditis Type Pericarditis",
    "pleura_effus_mention": "Pleural Effusion Mention",
    "hcm_mention": "Hypertrophic Cardiomyopathy Mention",
    "infiltrative_cm_mention": "Infiltrative Cardiomyopathy Mention",
    "infiltrative_cm_oth_type_not_oth": "Infiltrative Cardiomyopathy Other Type Not Otherwise Specified",
    "amyloid_mention": "Amyloidosis Mention",
    "sarcoid_mention": "Sarcoidosis Mention",
    "ischemic_cm_mention": "Ischemic Cardiomyopathy Mention",
    "nonischemic_cm_mention": "Nonischemic Cardiomyopathy Mention",
    "myocardial_infarct_mention": "Myocardial Infarction Mention",
    "hypertensive_pul_dis_mention": "Hypertensive Pulmonary Disease Mention"
}