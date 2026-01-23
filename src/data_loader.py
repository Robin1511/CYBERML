"""
Module pour charger et fusionner les données CICFlowMeter
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional


def load_all_csv_files(data_dir: str = "data", max_rows_per_file: Optional[int] = None, max_rows_per_attack_type: Optional[int] = None, sample_ratio: Optional[float] = None) -> pd.DataFrame:
    """
    Charge tous les fichiers CSV de chaque dossier et les fusionne en un seul DataFrame.
    Utilise un chargement optimisé en mémoire pour éviter les crashes.
    
    Args:
        data_dir: Chemin vers le dossier contenant les sous-dossiers de données
        max_rows_per_file: Nombre maximum de lignes à charger par fichier (None = tout charger)
        max_rows_per_attack_type: Nombre maximum de lignes par type d'attaque (None = pas de limite)
        sample_ratio: Ratio d'échantillonnage (0.0-1.0) pour limiter la taille totale (None = pas d'échantillonnage)
        
    Returns:
        DataFrame fusionné avec une colonne 'attack_type' ajoutée
    """
    import random
    data_path = Path(data_dir)
    all_chunks = []
    rows_per_attack = {}
    attack_folders = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
    
    print("Chargement des données (mode optimisé mémoire)...")
    
    for folder_name in attack_folders:
        folder_path = data_path / folder_name
        
        if not folder_path.exists():
            print(f"Attention: Le dossier {folder_name} n'existe pas, ignoré.")
            continue
        
        csv_files = list(folder_path.glob("**/*.csv"))
        
        if not csv_files:
            print(f"Attention: Aucun fichier CSV trouvé dans {folder_name}")
            continue
        
        print(f"  Chargement de {len(csv_files)} fichier(s) depuis {folder_name}...")
        rows_per_attack[folder_name] = 0
        if max_rows_per_attack_type and len(csv_files) > 1:
            max_files_to_use = min(len(csv_files), max(5, len(csv_files) // 2))
            rows_per_file = max_rows_per_attack_type // max_files_to_use
            files_to_use = csv_files[:max_files_to_use]
            print(f"    → Répartition: {max_files_to_use} fichier(s) avec ~{rows_per_file:,} lignes chacun")
        else:
            files_to_use = csv_files
            rows_per_file = None
        for csv_file in files_to_use:
            try:
                if max_rows_per_attack_type and rows_per_attack.get(folder_name, 0) >= max_rows_per_attack_type:
                    print(f"    ⊘ {csv_file.name}: limite atteinte pour {folder_name}")
                    continue
                remaining_for_type = max_rows_per_attack_type - rows_per_attack.get(folder_name, 0) if max_rows_per_attack_type else None
                target_for_this_file = min(rows_per_file, remaining_for_type) if (rows_per_file and remaining_for_type) else remaining_for_type
                
                chunk_size = 50000
                file_rows_loaded = 0
                
                for chunk in pd.read_csv(csv_file, low_memory=False, chunksize=chunk_size):
                    if max_rows_per_attack_type:
                        remaining = max_rows_per_attack_type - rows_per_attack.get(folder_name, 0)
                        if remaining <= 0:
                            break
                        if target_for_this_file and file_rows_loaded >= target_for_this_file:
                            break
                        if target_for_this_file:
                            max_for_chunk = min(remaining, target_for_this_file - file_rows_loaded)
                        else:
                            max_for_chunk = remaining
                        
                        if len(chunk) > max_for_chunk:
                            chunk = chunk.head(max_for_chunk)

                    if max_rows_per_file and len(chunk) > max_rows_per_file:
                        chunk = chunk.head(max_rows_per_file)

                    if sample_ratio and sample_ratio < 1.0:
                        if random.random() > sample_ratio:
                            continue
                    chunk['attack_type'] = folder_name
                    all_chunks.append(chunk)
                    chunk_rows = len(chunk)
                    rows_per_attack[folder_name] = rows_per_attack.get(folder_name, 0) + chunk_rows
                    file_rows_loaded += chunk_rows

                    if len(all_chunks) >= 10:
                        temp_df = pd.concat(all_chunks, ignore_index=True)
                        all_chunks = [temp_df]
                
                if file_rows_loaded > 0:
                    print(f"    {csv_file.name}: {file_rows_loaded:,} lignes chargées")
                else:
                    print(f"    {csv_file.name}: limite atteinte")
                
            except Exception as e:
                print(f"    Erreur lors du chargement de {csv_file.name}: {e}")
                continue
        
        if max_rows_per_attack_type and len(csv_files) > len(files_to_use):
            for csv_file in csv_files[len(files_to_use):]:
                print(f"    {csv_file.name}: limite atteinte pour {folder_name}")

    if not all_chunks:
        raise ValueError("Aucune donnée n'a pu être chargée!")
    
    print("\nFusion finale des chunks...")
    merged_df = pd.concat(all_chunks, ignore_index=True)
    
    print(f"\nTotal: {len(merged_df)} lignes fusionnées")
    print(f"Colonnes: {len(merged_df.columns)}")
    print(f"\nRépartition par type d'attaque:")
    for attack_type, count in rows_per_attack.items():
        print(f"  {attack_type}: {count:,} lignes")
    
    return merged_df


def check_column_consistency(dataframes: List[pd.DataFrame]) -> Dict:
    """
    Vérifie la cohérence des colonnes entre différents DataFrames.
    
    Args:
        dataframes: Liste de DataFrames à comparer
        
    Returns:
        Dictionnaire avec les informations de cohérence
    """
    if not dataframes:
        return {}
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)
    
    column_presence = {}
    for col in all_columns:
        column_presence[col] = []
        for i, df in enumerate(dataframes):
            if col in df.columns:
                column_presence[col].append(i)

    common_columns = [col for col, presence in column_presence.items() 
                     if len(presence) == len(dataframes)]
    
    missing_columns = {col: presence for col, presence in column_presence.items() 
                      if len(presence) < len(dataframes)}
    
    return {
        'all_columns': sorted(all_columns),
        'common_columns': sorted(common_columns),
        'missing_columns': missing_columns,
        'total_dataframes': len(dataframes)
    }


def balance_dataset(df: pd.DataFrame, target_benign_ratio: float = 0.5, balance_attack_types: bool = True, random_state: int = 42) -> pd.DataFrame:
    """
    Équilibre le dataset pour avoir un ratio spécifique entre Benign et attaques.
    
    Args:
        df: DataFrame avec colonne 'attack_type'
        target_benign_ratio: Ratio cible de trafic Benign (0.5 = 50% Benign, 50% attaques)
        balance_attack_types: Si True, équilibre aussi les différentes classes d'attaques entre elles
        random_state: Seed pour la reproductibilité
        
    Returns:
        DataFrame équilibré
    """
    import random
    random.seed(random_state)
    np.random.seed(random_state)
    
    df = df.copy()
    
    benign_df = df[df['attack_type'] == 'Benign'].copy()
    attacks_df = df[df['attack_type'] != 'Benign'].copy()
    
    n_benign = len(benign_df)
    n_attacks_total = len(attacks_df)
    
    print(f"\n{'='*60}")
    print("ÉQUILIBRAGE DU DATASET")
    print(f"{'='*60}")
    print(f"Avant équilibrage:")
    print(f"  Benign: {n_benign:,} lignes ({n_benign/(n_benign+n_attacks_total)*100:.1f}%)")
    print(f"  Attaques: {n_attacks_total:,} lignes ({n_attacks_total/(n_benign+n_attacks_total)*100:.1f}%)")
    
    if target_benign_ratio == 0.5:
        target_size = min(n_benign, n_attacks_total)
        target_benign = target_size
        target_attacks = target_size
    else:
        target_benign = int(n_benign * target_benign_ratio / (1 - target_benign_ratio)) if target_benign_ratio < 1.0 else n_benign
        target_attacks = int(n_attacks_total * (1 - target_benign_ratio) / target_benign_ratio) if target_benign_ratio > 0.0 else n_attacks_total
        target_size = min(target_benign, target_attacks)
        target_benign = target_size
        target_attacks = target_size
    
    if len(benign_df) > target_benign:
        benign_df = benign_df.sample(n=target_benign, random_state=random_state)
        print(f"\n  Benign échantillonné: {len(benign_df):,} lignes")
    else:
        print(f"\n  Benign conservé: {len(benign_df):,} lignes")
    
    if balance_attack_types:
        attack_types = attacks_df['attack_type'].unique()
        n_attack_types = len(attack_types)
        
        attack_type_counts = {at: len(attacks_df[attacks_df['attack_type'] == at]) for at in attack_types}
        
        ideal_per_type = target_attacks // n_attack_types
        remainder = target_attacks % n_attack_types
        
        print(f"\n  Équilibrage des {n_attack_types} types d'attaques (cible: {target_attacks:,} lignes total):")
        balanced_attacks = []
        total_collected = 0
        
        sorted_attacks = sorted(attack_type_counts.items(), key=lambda x: x[1])
        
        for i, (attack_type, n_available) in enumerate(sorted_attacks):
            attack_type_df = attacks_df[attacks_df['attack_type'] == attack_type]
            
            target_for_this_type = ideal_per_type + (1 if i < remainder else 0)
            
            take_count = min(target_for_this_type, n_available)
            
            if take_count > 0:
                sampled = attack_type_df.sample(n=take_count, random_state=random_state)
                balanced_attacks.append(sampled)
                total_collected += take_count
                print(f"    {attack_type}: {take_count:,} lignes (sur {n_available:,} disponibles)")
        
        if total_collected < target_attacks:
            remaining = target_attacks - total_collected
            print(f"\n  Complétant avec {remaining:,} lignes supplémentaires...")
            
            sorted_attacks_desc = sorted(attack_type_counts.items(), key=lambda x: x[1], reverse=True)
            
            for attack_type, n_available in sorted_attacks_desc:
                if remaining <= 0:
                    break
                
                existing_df = None
                existing_idx = None
                for idx, df_part in enumerate(balanced_attacks):
                    if len(df_part) > 0 and attack_type in df_part['attack_type'].values:
                        existing_df = df_part
                        existing_idx = idx
                        break
                
                if existing_df is not None:
                    current_count = len(existing_df)
                    if n_available > current_count:
                        original_df = df[df['attack_type'] == attack_type]
                        used_indices = set(existing_df.index) if hasattr(existing_df, 'index') else set()
                        available_df = original_df[~original_df.index.isin(used_indices)]
                        
                        if len(available_df) > 0:
                            additional_needed = min(remaining, n_available - current_count, len(available_df))
                            additional = available_df.sample(n=additional_needed, random_state=random_state)
                            balanced_attacks[existing_idx] = pd.concat([existing_df, additional], ignore_index=True)
                            remaining -= len(additional)
                            total_collected += len(additional)
                            print(f"    {attack_type}: +{len(additional):,} lignes supplémentaires (total: {len(balanced_attacks[existing_idx]):,})")
        
        attacks_df = pd.concat(balanced_attacks, ignore_index=True)
        
        if len(attacks_df) < target_attacks:
            remaining = target_attacks - len(attacks_df)
            print(f"\n  Complétant avec {remaining:,} lignes aléatoires depuis tous les types...")
            
            all_attacks_original = df[df['attack_type'] != 'Benign']
            used_indices = set(attacks_df.index) if hasattr(attacks_df, 'index') else set()
            available_attacks = all_attacks_original[~all_attacks_original.index.isin(used_indices)]
            
            if len(available_attacks) > 0:
                additional = available_attacks.sample(n=min(remaining, len(available_attacks)), random_state=random_state)
                attacks_df = pd.concat([attacks_df, additional], ignore_index=True)
                print(f"    Complété avec {len(additional):,} lignes aléatoires")
    else:
        if len(attacks_df) > target_attacks:
            attacks_df = attacks_df.sample(n=target_attacks, random_state=random_state)
            print(f"\n  Attaques échantillonnées: {len(attacks_df):,} lignes")
        else:
            print(f"\n  Attaques conservées: {len(attacks_df):,} lignes")
    
    balanced_df = pd.concat([benign_df, attacks_df], ignore_index=True)
    
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_benign_final = len(balanced_df[balanced_df['attack_type'] == 'Benign'])
    n_attacks_final = len(balanced_df[balanced_df['attack_type'] != 'Benign'])
    total_final = len(balanced_df)
    
    print(f"\nAprès équilibrage:")
    print(f"  Benign: {n_benign_final:,} lignes ({n_benign_final/total_final*100:.1f}%)")
    print(f"  Attaques: {n_attacks_final:,} lignes ({n_attacks_final/total_final*100:.1f}%)")
    print(f"  Total: {total_final:,} lignes")
    print(f"\nRépartition par type d'attaque:")
    print(balanced_df['attack_type'].value_counts().sort_index())
    print(f"{'='*60}\n")
    
    return balanced_df


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labellise automatiquement les données en remplaçant 'NeedManualLabel' 
    et en créant les colonnes isAttack et attack_category.
    
    Args:
        df: DataFrame avec colonne 'Label' contenant 'NeedManualLabel'
        
    Returns:
        DataFrame avec les colonnes de labels ajoutées
    """
    df = df.copy()
    
    if 'Label' in df.columns:
        mask = df['Label'] == 'NeedManualLabel'
        df.loc[mask, 'Label'] = df.loc[mask, 'attack_type']
    
    df['isAttack'] = (df['attack_type'] != 'Benign').astype(int)
    
    df['attack_category'] = df['attack_type']
    
    print("\nDistribution des labels:")
    print(df['attack_type'].value_counts().sort_index())
    print(f"\nTotal Benign: {(df['isAttack'] == 0).sum()}")
    print(f"Total Attacks: {(df['isAttack'] == 1).sum()}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Chargement et labellisation des données CICFlowMeter")
    print("=" * 60)
    
    df = load_all_csv_files()
    
    df = label_data(df)
    
    print("\n" + "=" * 60)
    print("Informations sur le dataset:")
    print("=" * 60)
    print(f"Nombre de lignes: {len(df)}")
    print(f"Nombre de colonnes: {len(df.columns)}")
    print(f"\nPremières colonnes: {list(df.columns[:10])}")
    print(f"\nTypes de données:")
    print(df.dtypes.value_counts())
    

