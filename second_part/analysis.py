#!/usr/bin/env python3.12
# coding=utf-8

#%%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile


# Ukol 1: nacteni dat ze ZIP souboru

def load_data(filename: str, ds: str) -> pd.DataFrame:
    """
    Načtení dat z daného ZIP souboru, spojí tabulky podle datové sady (ds).

    :param filename: Jméno ZIP souboru
    :param ds: Klíč pro hledání datových souborů v ZIP
    :return: Sloučená tabulka jako DataFrame
    """
    with zipfile.ZipFile(filename, 'r') as z:
        files_to_concatenate = [f for f in z.namelist() if ds in f]
        dfs = []
        for file in files_to_concatenate:
            with z.open(file) as xls_file:
                df = pd.read_html(xls_file, encoding='cp1250')[0]
                df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                dfs.append(df_cleaned)
        return pd.concat(dfs, ignore_index=True)

# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parsování a čištění dat, převod datumu, regionů a odstranění duplicit.

    :param df: Vstupní DataFrame
    :param verbose: Flag pro tisk paměťové náročnosti
    :return: Upravený DataFrame
    """
    reg_codes = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK", 6: "JHM", 7: "MSK",
        14: "OLK", 15: "ZLK", 16: "VYS", 17: "PAK", 18: "LBK", 19: "KVK"
    }
    parsed_df = df.copy()
    parsed_df['datum'] = pd.to_datetime(parsed_df['p2a'], format='%d.%m.%Y')
    parsed_df['region'] = parsed_df['p4a'].replace(reg_codes)
    parsed_df = parsed_df.drop_duplicates(subset='p1', keep='first')

    if verbose:
        print(f"Paměťová náročnost: {parsed_df.memory_usage(deep=True).sum() / 10 ** 6:.1f} MB")

    return parsed_df

# Ukol 3: Počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Grafy znázorně ného stavu vozovky a jejich vliv na nehody podle regionů.

    :param df: DataFrame s daty o nehodách
    :param fig_location: Cesta k uložení grafu
    :param show_figure: Zda zobrazit graf
    """
    road_conditions = {
        1: "povrch suchý", 2: "povrch suchý", 3: "povrch mokrý", 
        4: "na vozovce je bláto", 
        5: "na vozovce je náledí, ujetý sníh", 
        6: "na vozovce je náledí, ujetý sníh"
    }

    df['road_condition'] = df['p16'].where(df['p16'].between(1, 6), np.nan)
    df = df.replace({'road_condition': road_conditions})
    incident_counts = df.groupby(['road_condition', 'region']).size().unstack(fill_value=0)

    colors = sns.color_palette("tab10", len(road_conditions))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (condition, counts) in enumerate(incident_counts.iterrows()):
        sns.barplot(
            x=counts.index, y=counts.values, ax=axes[i], color=colors[i]
        )
        axes[i].grid(which='major', axis='y', linestyle='--', linewidth=0.8)
        axes[i].set_facecolor('#F5F5F5')
        axes[i].set_axisbelow(True)
        axes[i].set_title(f'Stav vozovky: {condition}')
        axes[i].set_ylabel('Počet nehod')

    if fig_location:
        fig.savefig(fig_location, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close()

# Ukol 4: Alkohol a následky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Grafy vlivu alkoholu na následky nehod v jednotlivých regionech.

    :param df: DataFrame s daty o nehodách
    :param df_consequences: DataFrame s daty o následcích
    :param fig_location: Cesta k uložení grafu
    :param show_figure: Zda zobrazit graf
    """
    injury_dict = {1: "usmrceni", 2: "těžké zranění", 3: "lehké zranění", 4: "bez zranění"}

    alcohol_df = pd.merge(df, df_consequences, on='p1')
    alcohol_df = alcohol_df[alcohol_df['p11'] >= 3]
    alcohol_df = alcohol_df.replace({"p59g": injury_dict})

    alcohol_driver = alcohol_df[alcohol_df['p59a'] == 1]
    alcohol_passenger = alcohol_df[alcohol_df['p59a'] != 1]

    #Calculate amount of accidents with driver and passenger
    driver_incidents = alcohol_driver.groupby(['p59g', 'region']).size().unstack(fill_value=0)
    passenger_incidents = alcohol_passenger.groupby(['p59g', 'region']).size().unstack(fill_value=0)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axs = axs.flatten()

    for i, ((index1, row1), (index2, row2)) in enumerate(zip(driver_incidents.iterrows(), passenger_incidents.iterrows())):
        combined_data = pd.concat([row1, row2], axis=1)
        combined_data.plot.bar(ax=axs[i], color=['steelblue', 'tomato'], legend=False)
        axs[i].set_title(f'Následky: {index1}')
        axs[i].set_axisbelow(True)
        axs[i].set_ylabel('Počet nehod pod vlivem')
        axs[i].grid(which='major', axis='y', linestyle='--', linewidth=0.8)

    handles = [
        plt.Line2D([0], [0], color='steelblue', lw=4, label='Řidič'),
        plt.Line2D([0], [0], color='tomato', lw=4, label='Spolujezdec')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if fig_location:
        fig.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        plt.show()
    else:
        plt.close()
        

# Ukol 5: Druh nehody (srážky) v čase
def plot_type(df: pd.DataFrame, fig_location: str = None,
              show_figure: bool = False):
    
    """
    Plot the type of accidents over time for selected regions.

    :param df: A DataFrame containing accident data.
    :param fig_location: The file path to save the figure (optional).
    :param show_figure: If True, displays the plot.
    """
    
    accident_type= {1:"s jedoucím nekolejovým vozidlem", 2: "s vozidlem zaparkovaným, odstaveným", 3: "s pevnou překážkou", 
                    4: "s chodcem", 5: "s lesní zvěří", 6: "s domácím zvířetem", 7: "s vlakem", 8: "s tramvají"}
    
    #Choosed regions
    selected_regions = ["KVK", "OLK", "STC", "ZLK"]
    typeDF = df[(df['region'].isin(selected_regions))]
    typeDF = typeDF[typeDF['p6'].between(1,8)]
    typeDF = typeDF.replace({'p6': accident_type})
    
    typeDF["month"] = typeDF["datum"].dt.to_period("M")
    
    #Calculate amount of accidents by region and month
    monthly_summary = typeDF.groupby(["month", "region", "p6"]).size().reset_index(name="count")
    monthly_summary['month'] = monthly_summary['month'].dt.to_timestamp()

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", len(accident_type))
    hue_order = list(accident_type.values())
    
    for i,region in enumerate(selected_regions):
        region_data = monthly_summary[monthly_summary['region'] == region]
        
        sns.lineplot(
            data=region_data,
            x="month",
            y="count",
            hue="p6",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            ax=axes[i]
        )
        
        axes[i].set_title(f"Region {region}")
        axes[i].set_xlabel("Datum")
        axes[i].set_ylabel("Počet nehod")
        axes[i].grid(True)

        # Odebrání jednotlivých legend
        axes[i].get_legend().remove()
        
    fig.legend(
        title="Druh nehody",
        labels=hue_order,
        loc='upper center',
        ncol=4,
        fontsize=10
    )
    
    plt.xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-09"))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
        
if __name__ == "__main__":
    # Define the ZIP file and dataset name
    zip_filename = "data_23_24.zip"  # Replace with your actual ZIP file

    # Load data
    df = load_data(zip_filename, "nehody")

    # Parse data
    df = parse_data(df, verbose=True)

    # Generate and display/save plots
    plot_state(df, fig_location="plot_state.png", show_figure=True)

    df_consequences = load_data(zip_filename, "nasledky")

    plot_alcohol(df, df_consequences, fig_location="plot_alcohol.png", show_figure=True)

    plot_type(df, fig_location="plot_type.png", show_figure=True)
