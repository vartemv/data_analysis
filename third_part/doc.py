import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import re
from typing import TextIO

# Define categories of accidents with their corresponding ranges of codes
categories = {
    "not_drivers_fault": [100],
    "speed_problem": range(201, 210),
    "wrong_bypass": range(301, 312),
    "didnt_give_prednost": range(401, 415),
    "nespravny_zpusob_jezdy": range(501, 517),
    "technical_problem_with_car": range(601, 616),
}

def find_most_common_number_by_category(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """
    Find the most common number in each category and its percentage.
    :param df: DataFrame to examine
    :param categories: Dictionary defining category ranges
    :return: DataFrame summarizing the most common number and its percentage for each category
    """
    results = []

    for category, value_range in categories.items():
        # Filter data for the current category
        mask = df['p12'].isin(value_range)
        category_data = df[mask]

        if not category_data.empty:
            # Find the most common number and its count
            most_common_number = category_data['p12'].mode().iloc[0]
            most_common_count = category_data['p12'].value_counts().iloc[0]

            # Calculate the percentage
            total_count = len(category_data)
            percentage = (most_common_count / total_count) * 100

            # Store the result
            results.append({
                "Category": category,
                "Most Common Reason": most_common_number,
                "Count": most_common_count,
                "Percentage": percentage
            })

    # Convert results to DataFrame
    summary_df = pd.DataFrame(results)
    return summary_df

def create_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table summarizing the number of accidents of each reason type for years 2023 and 2024.
    :param df: DataFrame to examine
    :return: Formatted DataFrame
    """

    # Group data by reason and year
    data = df.groupby([df['reason'], df['date'].dt.year]).size().unstack(fill_value=0)

    # Rename columns to match years
    data.columns = ['2023', '2024']
    data.index.name = 'Accident Reason'

    # Return the formatted DataFrame
    return data

def table_to_tex(df: pd.DataFrame, stream: TextIO = sys.stdout):
    """
    Print the formatted table as LaTeX output with dividers.
    :param df: DataFrame to convert
    :param stream: Stream to write the data
    :return: None
    """
    tex = df.to_latex(
        caption="Accidents by Reason in 2023 and 2024",
        position="h",
        bold_rows=True,
        column_format="|l|" + "c|" * len(df.columns),  # Add dividers for all columns
        index_names=True,  # Include index name in the LaTeX table
        multicolumn=False,  # Avoid multicolumn formatting
    )

    # Add horizontal dividers (hline) around the header and footer
    tex = re.sub(r"\\begin{tabular}{.*}\n", r"\\begin{tabular}{|" + "l|" + "c|" * len(df.columns) + r"}\n\\hline\n", tex, flags=re.MULTILINE)
    tex = re.sub(r"\\end{tabular}", r"\\hline\n\\end{tabular}", tex, flags=re.MULTILINE)
    tex = re.sub(r"\\toprule", r"\\hline", tex, flags=re.MULTILINE)
    tex = re.sub(r"\\midrule", r"\\hline", tex, flags=re.MULTILINE)
    tex = re.sub(r"\\bottomrule", r"\\hline", tex, flags=re.MULTILINE)

    # Print the LaTeX table
    print("%%%%%%%% LATEX TABLE %%%%%%%%")
    print(tex, file=stream, end="")
    print("%%%%%%%% LATEX TABLE %%%%%%%%")

def map_reason(value):
    """
    Assign a description of the accident reason based on the p12 value.
    :param value: Value in the p12 column
    :return: String describing the accident reason
    """
    if value == 100:
        return 'Not the drivers fault'
    elif 201 <= value <= 209:
        return 'Speed-related issue'
    elif 301 <= value <= 311:
        return 'Incorrect overtaking maneuver'
    elif 401 <= value <= 414:
        return 'Failure to yield'
    elif 501 <= value <= 516:
        return 'Incorrect driving technique'
    elif 601 <= value <= 615:
        return 'Technical issue with the vehicle'
    else:
        return 'Other reason'

def get_dataframe():
    """
    Load the data and add a reason column based on mapping logic.
    :return: DataFrame with mapped reasons
    """
    df = pd.read_pickle("accidents.pkl.gz")  # Load data from pickle file
    df['reason'] = df['p12'].apply(map_reason)  # Map p12 values to reason descriptions
    return df

def plot_graphs(dataframe: pd.DataFrame, fig_location: str, show_figure: bool):
    """
    Create and save or display graphs for accident data.
    :param dataframe: DataFrame with accident data
    :param fig_location: File path to save the figure
    :param show_figure: Whether to display the figure
    """
    # Group data by region and reason
    data = dataframe.groupby(['region', 'reason']).size().reset_index(name='count')

    # Sum totals for pie chart
    total_reasons = dataframe['reason'].value_counts()

    # Set up plot styles
    sns.set_theme(style="whitegrid")
    color_palette = sns.color_palette("tab10", len(total_reasons))  # Consistent color palette

    # Map colors to reasons
    reason_colors = {reason: color for reason, color in zip(total_reasons.index, color_palette)}

    # Create a figure with a grid layout
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Increase font sizes globally
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=18)

    # Bar plot - Regional distribution of reasons
    ax1 = fig.add_subplot(spec[1, :])
    sns.barplot(data=data, x='region', y='count', hue='reason', palette=reason_colors, ax=ax1, legend=False)
    ax1.set_title("Regional Distribution of Accident Reasons")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Number of Accidents")

    # Pie chart - Distribution of accident reasons overall
    ax2 = fig.add_subplot(spec[0, :])
    total_reasons.plot(
        kind="pie",
        autopct=lambda pct: ('%.1f%%' % pct) if pct > 5 else '',  # Show percentages greater than 5%
        ax=ax2,
        colors=[reason_colors[reason] for reason in total_reasons.index],  # Use same colors
        legend=False,
        labels=None
    )
    ax2.set_title("Overall Distribution of Accident Reasons")
    ax2.set_ylabel("")  # Remove y-axis label for the pie chart
    ax2.legend(total_reasons.index, title="Reasons", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save or show the figure
    if fig_location:
        Path(fig_location).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        plt.savefig(fig_location)  # Save the figure

    if show_figure:
        plt.show()  # Display the figure

if __name__ == '__main__':
    # Load the data
    df = get_dataframe()

    # Generate and display/save graphs
    plot_graphs(df, fig_location='fig.pdf', show_figure=True)

    # Create and output the LaTeX table
    table_to_tex(create_table(df))

    # Separator for readability
    print(""" 
#########  
          """)

    # Find and print the most common number by category
    print(find_most_common_number_by_category(df, categories))