DATASET_CONFIGS = {
    "limca_hg38": {
        "metadata_path": "/public/home/genomics3d/zyc/coHREmbed/data/LimCa/celllinehg38/limcahg38_labels.csv",
        "label_columns": ["Cell type"],
        
    "hic_path":"/public/home/genomics3d/zyc/coHREmbed/data/LimCa/celllinehg38/limca_cellline_hg38_1mb_sep_noimpute.pkl",
        "rna_path": "/public/home/genomics3d/zyc/coHREmbed/data/LimCa/celllinehg38/GSE240114_cellline_umi_counts.csv",
        "n_cluster": {
            "Cell type": 4,
        },
        "output_dir": "./limcahg38",
        "hic_resolution": "1mb",
        "epochs": 100,
    }, 
    
    "limca_osn": {
        "metadata_path": "/public/home/genomics3d/zyc/coHREmbed/data/LimCa/osnmm10/limca-osn-label.csv",
        "label_columns": ["Cluster"],
        'hic_path': "/public/home/genomics3d/zyc/coHREmbed/data/LimCa/osnmm10/limca_osn_mm10_1mb_sep_noimpute.pkl",
        'rna_path': "/public/home/genomics3d/zyc/coHREmbed/data/LimCa/osnmm10/GSE240114_moe_umi_counts.csv",
        'n_cluster' : {
            "Cluster":5
        },
        "output_dir": "./limca_osn",
        "hic_resolution": "1mb",
        "epochs": 100,
    },
    
    "hires_brain": {
        "metadata_path": "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/GSE223917_HiRES_brain_metadata.csv",
        "label_columns": ["Cell type"],
        "hic_path": "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/hires-brain-mm10/Hires_brain_cnn_1mb_sep_noimpute.pkl",
        "rna_path": "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/GSE223917_HiRES_brain.rna.umicount.tsv",
        "n_cluster": {
            "Cell type": 7,
        },
        "output_dir": "./hires_brain",
        "hic_resolution": "1mb",
        "epochs": 100,
    },

    
    "hires_mesc": {
        "metadata_path": "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/GSE223917_HiRES_emb_metadata.csv",
        "label_columns": ["Celltype"],
        'hic_path': "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/hires-mESC-mm10/Hires_mESC_cnn_1mb_sep_noimpute.pkl",
        'rna_path': "/public/home/genomics3d/zyc/coHREmbed/data/HiRES/GSE223917_HiRES_emb.rna.umicount.tsv",
        'n_cluster': {
            "Celltype": 21
        },
        "output_dir": "./hires_mesc",
        "hic_resolution": "1mb",
        "epochs": 200,
    },

    
    "gage_mm10": {
        "metadata_path": "/public/home/genomics3d/zyc/coHREmbed/data/Gage/mbcmm10/meta-Seurat_mBC_complete_mm10_filtered_with_Celltype24.csv",
        "label_columns": ["Cell type24"],
        'hic_path': "/public/home/genomics3d/zyc/coHREmbed/data/Gage/mbcmm10/gage_mm10_1mb_sep_noimpute.pkl",
        'rna_path': "/public/home/genomics3d/zyc/coHREmbed/data/Gage/mbcmm10/gage_mbc_umicount.csv",
        'n_cluster': {"Cell type24": 24},
        "output_dir": "./gage_mm10",
        "hic_resolution": "1mb",
        "epochs": 200,
    },

    
    "gage_bone": {
        "metadata_path": "/root/private_data/zyc/coHREmbed/data/Gage/cdhg38/meta-python_CD_complete_hg38_filtered_with_Celltype8.csv",
        "label_columns": ["Celltype8"],
        'hic_path': "/root/private_data/zyc/coHREmbed/data/Gage/cdhg38/gage_hg38_1mb_sep_noimpute.pkl",
        'rna_path': "/root/private_data/zyc/coHREmbed/data/Gage/cdhg38/gage_bone_umicount.csv",
        'n_cluster': {"Celltype8": 8},
        "output_dir": "./gage_bone",
        "hic_resolution": "1mb",
        "epochs": 100,
    },
}


def get_dataset_config(name):
    if name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return DATASET_CONFIGS[name].copy()
