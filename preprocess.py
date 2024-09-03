import pandas as pd
import os
import json 

######################
######################
# I. AIOE_MultiModal #
######################
######################

##############
# O*NET data #
##############

# O*NET Abilities data
abilities = pd.read_excel(f"""{os.path.join("data", "ONET_Abilities_v24-2.xlsx")}""")
abilities.drop(
    columns=[
        "Title",
        "Element ID",
        "N",
        "Standard Error",
        "Lower CI Bound",
        "Upper CI Bound",
        "Recommend Suppress",
        "Not Relevant",
        "Date",
        "Domain Source",
        "Scale Name",
    ],
    inplace=True,
)

# Collapse 8-digit SOC code to 6-digit SOC code
# 1. Create a new column with the six-digit SOC code
abilities["Six-Digit SOC Code"] = abilities["O*NET-SOC Code"].str[:7]
# 2. Group the DataFrame by the new SOC code, Element Name, and Scale ID
grouped = abilities.groupby(
    ["Six-Digit SOC Code", "Element Name", "Scale ID"], as_index=False
)
# 3. Calculate the mean of the Data Value for each group
abilities_condensed = grouped["Data Value"].mean()

##############################################################
# Application-Ability Matrix (Felten, Raj and Seamans, 2021) #
##############################################################

aa_matrix = pd.read_excel(
    f"""{os.path.join("AIOE", "AIOE_DataAppendix.xlsx")}""", sheet_name="Appendix D"
)
aa_matrix = aa_matrix.rename(columns={"Unnamed: 0": "Element Name"})

# 1.Specify the columns to sum
columns_to_sum = [
    # 'Abstract Strategy Games', 'Real-Time Video Games',
    "Image Recognition",
    "Visual Question Answering",
    "Generating Images",
    "Reading Comprehension",
    "Language Modeling",
    "Translation",
    "Speech Recognition",
    "Instrumental Track Recognition",
]

# 2. Calculate the row-wise sum and create the new column 'MultiModal_AbilityLevelExposure'
aa_matrix["MultiModal_AbilityLevelExposure"] = aa_matrix[columns_to_sum].sum(axis=1)
aa_matrix.drop(
    columns=[
        "Abstract Strategy Games",
        "Real-Time Video Games",
        "Image Recognition",
        "Visual Question Answering",
        "Generating Images",
        "Reading Comprehension",
        "Language Modeling",
        "Translation",
        "Speech Recognition",
        "Instrumental Track Recognition",
    ],
    inplace=True,
)
aa_matrix.loc[
    aa_matrix["Element Name"] == "Visual Color Determination", "Element Name"
] = "Visual Color Discrimination" 

# 3. Merge O*NET data with Application-Ability Matrix
merged_df = abilities_condensed.merge(aa_matrix, how="left", on="Element Name")
transformed_df = merged_df.pivot_table(
    index=["Six-Digit SOC Code", "Element Name", "MultiModal_AbilityLevelExposure"],
    columns="Scale ID",
    values="Data Value",
).reset_index()
transformed_df = transformed_df.rename_axis(None, axis=1).reset_index(drop=True)

# 4. Group by 'Six-Digit SOC Code' and calculate AIOE_MultiModal for each SOC code
AIOE_MultiModal = (
    transformed_df.groupby("Six-Digit SOC Code")
    .apply(
        lambda group: (
            (group["MultiModal_AbilityLevelExposure"] * group["IM"] * group["LV"]).sum()
            / (group["IM"] * group["LV"]).sum()
        )
    )
    .reset_index(name="AIOE")
)
AIOE_MultiModal = AIOE_MultiModal.rename(columns={"Six-Digit SOC Code": "SOC Code"})

# 5. Standardize AIOE_MultiModal values
mean_AIOE = AIOE_MultiModal["AIOE"].mean()
std_AIOE = AIOE_MultiModal["AIOE"].std()
AIOE_MultiModal["Standardized AIOE"] = (AIOE_MultiModal["AIOE"] - mean_AIOE) / std_AIOE

###############################################
###############################################
# II. Occupation-to-Subreddit Mapping Dataset #
###############################################
###############################################

########################################
# SOC-to-Subreddit Mapping data (cite) #
########################################

df = pd.read_csv(f"""{os.path.join("data", "SOC-to-Subreddit-Mapping.csv")}""")

###################################################
# AIOE data (Felten, Raj and Seamans, 2021; 2023) #
###################################################

# AIOE_General (Felten, Raj and Seamans, 2021)
aioe_general = pd.read_excel(
    f"""{os.path.join("AIOE", "AIOE_DataAppendix.xlsx")}""", sheet_name="Appendix A"
)
# AIOE_Language (Felten, Raj and Seamans, 2023)
aioe_language = pd.read_excel(
    f"""{os.path.join("AIOE", "Language Modeling AIOE and AIIE.xlsx")}""",
    sheet_name="LM AIOE",
)
# AIOE_Image (Felten, Raj and Seamans, 2023)
aioe_image = pd.read_excel(
    f"""{os.path.join("AIOE", "Image Generation AIOE and AIIE.xlsx")}""",
    sheet_name="IG AIOE",
)

# Merge df with AIOE_General to add "AIOE_General"
df = df.merge(
    aioe_general[["SOC Code", "AIOE"]], how="left", left_on="SOC", right_on="SOC Code"
)
df = df.rename(columns={"AIOE": "AIOE_General"})
df = df.drop(columns=["SOC Code"])

# Merge df with AIOE_Language to add "AIOE_LanguageModel"
df = df.merge(
    aioe_language[["SOC Code", "Language Modeling AIOE"]],
    how="left",
    left_on="SOC",
    right_on="SOC Code",
)
df = df.rename(columns={"Language Modeling AIOE": "AIOE_LanguageModel"})
df = df.drop(columns=["SOC Code"])

# Merge df with AIOE_Image to add "AIOE_ImageModel"
df = df.merge(
    aioe_image[["SOC Code", "Image Generation AIOE"]],
    how="left",
    left_on="SOC",
    right_on="SOC Code",
)
df = df.rename(columns={"Image Generation AIOE": "AIOE_ImageModel"})
df = df.drop(columns=["SOC Code"])

# Merge df with AIOE_MultiModal to add "AIOE_MultiModal"
df = df.merge(
    AIOE_MultiModal[["SOC Code", "Standardized AIOE"]],
    how="left",
    left_on="SOC",
    right_on="SOC Code",
)
df = df.rename(columns={"Standardized AIOE": "AIOE_MultiModal"})
df = df.drop(columns=["SOC Code"])

# Reorder columns 
columns_order = [
    "SOC",
    "Occupation",
    "AIOE_General",
    "AIOE_LanguageModel",
    "AIOE_ImageModel",
    "AIOE_MultiModal",
] + [
    col
    for col in df.columns
    if col
    not in [
        "SOC",
        "Occupation",
        "AIOE_General",
        "AIOE_LanguageModel",
        "AIOE_ImageModel",
        "AIOE_MultiModal",
    ]
]
df = df[columns_order]

# Save df to csv 
df.to_csv("Occupations-to-Subreddits.csv")

####################################################
# Creating Occupation-to-Subreddit Mapping Dataset #
####################################################

from SOC_index import soc_index, soc_hierarchy_index

def calculate_summary(samples):
    import numpy as np
    summary = {
        "sample": samples,
        "mean": np.mean(samples),
        "median": np.median(samples),
        "std": np.std(samples),
        "minmax": [np.min(samples), np.max(samples)]
    }
    return summary

# 1. Create Basic SOC_dict 

soc_dict = {}

subreddit_columns = [col for col in df.columns if col.startswith('sub_')]

for _, row in df.iterrows():
    major_group, minor_group = row['SOC'].split('-')
    major_group = int(major_group)
    minor_group = int(minor_group)
    
    if major_group not in soc_dict:
        soc_dict[major_group] = {}
    
    if 'subreddit' not in soc_dict[major_group]:
        soc_dict[major_group]['subreddit'] = []
    
    for sub_key in subreddit_columns:
        if pd.notna(row[sub_key]) and row[sub_key] not in soc_dict[major_group]['subreddit']:
            soc_dict[major_group]['subreddit'].append(row[sub_key])
    
    soc_dict[major_group][minor_group] = {
        'occupation': row['Occupation'],
        'exposure': {
            "AIOE_General": row['AIOE_General'],
            "AIOE_Language": row['AIOE_LanguageModel'],
            "AIOE_Image": row['AIOE_ImageModel'], 
            "AIOE_MultiModal": round(row['AIOE_MultiModal'], 6)
        },
        'subreddit': []
    }
    
    for sub_key in subreddit_columns:
        if pd.notna(row[sub_key]):
            soc_dict[major_group][minor_group]['subreddit'].append(row[sub_key])

for major_group in soc_dict.keys():
    if not soc_dict[major_group]['subreddit']:
        soc_dict[major_group]['subreddit'] = [None]
    for minor_group in soc_dict[major_group].keys():
        if minor_group != 'subreddit' and not soc_dict[major_group][minor_group]['subreddit']:
            soc_dict[major_group][minor_group]['subreddit'] = [None]

# 2. Calculate summary info for SOC Major Groups           

for major_group, minor_group in soc_dict.items():
    if 'subreddit' in minor_group:
        del minor_group['subreddit']
    
    all_aioe_general_values = list(row['exposure']['AIOE_General'] for row in minor_group.values())
    all_aioe_language_values = list(row['exposure']['AIOE_Language'] for row in minor_group.values())
    all_aioe_image_values = list(row['exposure']['AIOE_Image'] for row in minor_group.values())
    all_aioe_multimodal_values = list(row['exposure']['AIOE_MultiModal'] for row in minor_group.values())
    
    unique_subreddits = set()
    for _, value in minor_group.items():
        unique_subreddits.update(value['subreddit'])
    unique_subreddits = [subreddit for subreddit in unique_subreddits if subreddit is not None]
    
    soc_dict[major_group]['summary'] = {
        'occupation': soc_index.get(major_group, 'Unknown'),
        'exposure': {
            'AIOE_General': calculate_summary(all_aioe_general_values),
            'AIOE_Language': calculate_summary(all_aioe_language_values), 
            'AIOE_Image': calculate_summary(all_aioe_image_values),
            'AIOE_MultiModal': calculate_summary(all_aioe_multimodal_values)
        },
        'subreddit': list(unique_subreddits)
    }

# 3. Sort keys in dictionary  
    
outer_sorted_soc_dict = {k: soc_dict[k] for k in sorted(soc_dict)}

for key in outer_sorted_soc_dict:
    if isinstance(outer_sorted_soc_dict[key], dict):
        inner_dict = outer_sorted_soc_dict[key]
        summary = inner_dict.pop("summary", None)
        sorted_inner_dict = dict(sorted(inner_dict.items(), key=lambda x: int(x[0])))
        if summary:
            sorted_inner_dict["summary"] = summary
        outer_sorted_soc_dict[key] = sorted_inner_dict
        
# 4. Calculate summary info for SOC Minor Groups 

ormd = {}

for soc_major, soc_minor in soc_hierarchy_index.items():
    ormd[soc_major] = {}
    for soc_minor_code, soc_minor_name in soc_minor.items():
        soc_minor_dict = {}
        aioe_general_samples, aioe_language_samples, aioe_image_samples, aioe_multi_samples = [], [], [], []
        subreddits = set()

        for occ_code, occ_info in outer_sorted_soc_dict.get(soc_major, {}).items():
            if str(occ_code).startswith(str(soc_minor_code)[:1]):
                soc_minor_dict[occ_code] = occ_info
                aioe_general_samples.append(occ_info["exposure"]["AIOE_General"])
                aioe_language_samples.append(occ_info["exposure"]["AIOE_Language"])
                aioe_image_samples.append(occ_info["exposure"]["AIOE_Image"])
                aioe_multi_samples.append(occ_info["exposure"]["AIOE_MultiModal"])
                subreddits.update(occ_info["subreddit"])

        if aioe_general_samples:
            soc_minor_dict["summary"] = {
                "occupation": soc_minor_name,
                "exposure": {
                    "AIOE_General": calculate_summary(aioe_general_samples), 
                    "AIOE_Language": calculate_summary(aioe_language_samples), 
                    "AIOE_Image": calculate_summary(aioe_image_samples),
                    "AIOE_MultiModal": calculate_summary(aioe_multi_samples),  
                }, 
                "subreddit": [subreddit for subreddit in list(subreddits) if subreddit is not None]
            }
            
        ormd[soc_major][soc_minor_code] = soc_minor_dict

    ormd[soc_major]["summary"] = outer_sorted_soc_dict[soc_major]["summary"]

# 5. Create condensed ORMD 

ormd_summary = {} 

for major_key, major_value in ormd.items():
    ormd_summary[major_key] = {}
    for minor_key, minor_value in major_value.items(): 
        for key, value in minor_value.items(): 
            if key == "summary": 
                ormd_summary[major_key][minor_key] = value
    ormd_summary[major_key]["occupation"] = ormd[major_key]["summary"]["occupation"]
    ormd_summary[major_key]["exposure"] = ormd[major_key]["summary"]["exposure"]
    ormd_summary[major_key]["subreddit"] = ormd[major_key]["summary"]["subreddit"]

# Save files 
with open('ORMD.json', 'w') as json_file:
    json.dump(ormd, json_file, indent=4)

with open('ORMD_condensed.json', 'w') as json_file:
    json.dump(ormd_summary, json_file, indent=4)