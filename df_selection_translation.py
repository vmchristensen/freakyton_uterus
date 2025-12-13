import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMS.csv"
df = pd.read_csv(file_path)

# Relevant columns (original names)
relevant_cols = [
    'FN', 'edad', 'imc', 'f_diag', 'tipo_histologico', 'Grado', 'valor_de_ca125',
    'infiltracion_mi', 'metasta_distan', 'grupo_riesgo', 'tto_NA', 'tto_1_quirugico',
    'asa', 'histo_defin', 'grado_histologi', 'tamano_tumoral', 'afectacion_linf',
    'AP_centinela_pelvico', 'AP_ganPelv', 'AP_glanPaor', 'recep_est_porcent', 'rece_de_Ppor',
    'beta_cateninap', 'estudio_genetico_r01', 'FIGO2023', 'grupo_de_riesgo_definitivo',
    'Tributaria_a_Radioterapia', 'Tratamiento_RT', 'Tratamiento_sistemico',
    'Ultima_fecha', 'recidiva', 'recidiva_exitus', 'diferencia_dias_reci_exit',
    'estado', 'causa_muerte', 'f_muerte', 'libre_enferm', 'numero_de_recid',
    'fecha_de_recidi', 'dx_recidiva', 'tto_recidiva', 'Tt_recidiva_qx',
    'Reseccion_macroscopica_complet','qt'
]

# Subset the dataframe
df_relevant = df[relevant_cols]

# Rename columns to English
rename_mapping = {
    'FN': 'birth_date',
    'edad': 'age',
    'imc': 'BMI',
    'f_diag': 'diagnosis_date',
    'tipo_histologico': 'histologic_type',
    'Grado': 'grade',
    'valor_de_ca125': 'CA125_value',
    'infiltracion_mi': 'myometrial_invasion',
    'metasta_distan': 'distant_metastasis',
    'grupo_riesgo': 'risk_group_preSurgery',
    'tto_NA': 'neoadjuvant_treatment',
    'tto_1_quirugico': 'first_surgery_treatment',
    'asa': 'ASA_score',
    'histo_defin': 'final_histology',
    'grado_histologi': 'histology_grade',
    'tamano_tumoral': 'tumor_size',
    'afectacion_linf': 'lymph_node_involvement',
    'AP_centinela_pelvico': 'pelvic_sentinel_nodes',
    'AP_ganPelv': 'pelvic_lymph_nodes',
    'AP_glanPaor': 'para_aortic_nodes',
    'recep_est_porcent': 'estrogen_receptors_pct',
    'rece_de_Ppor': 'progesterone_receptors',
    'beta_cateninap': 'beta_catenin',
    'estudio_genetico_r01': 'genetic_study_1',
    'FIGO2023': 'FIGO2023_stage',
    'grupo_de_riesgo_definitivo': 'definitive_risk_group',
    'Tributaria_a_Radioterapia': 'eligible_for_radiotherapy',
    'Tratamiento_RT': 'radiotherapy_treatment',
    'Tratamiento_sistemico': 'systemic_treatment',
    'Ultima_fecha': 'last_visit_date',
    'recidiva': 'recurrence',
    'recidiva_exitus': 'recurrence_death',
    'diferencia_dias_reci_exit': 'days_to_recurrence_or_death',
    'estado': 'current_status',
    'causa_muerte': 'cause_of_death',
    'f_muerte': 'death_date',
    'libre_enferm': 'disease_free',
    'numero_de_recid': 'recurrence_number',
    'fecha_de_recidi': 'recurrence_date',
    'dx_recidiva': 'recurrence_dx',
    'tto_recidiva': 'recurrence_treatment',
    'Tt_recidiva_qx': 'recurrence_surgery',
    'Reseccion_macroscopica_complet': 'complete_macroscopic_resection',
    'qt': 'chemotherapy'
}

df_relevant.rename(columns=rename_mapping, inplace=True)

# Save CSV with English column names
df_relevant.to_csv(r"C:\Users\nadine jager\Documents\hackathon\freakyton_uterus\IQ_Cancer_Endometrio_merged_NMS_relevant_english.csv", index=False)

# Print first rows
print(df_relevant.head())

