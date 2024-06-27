import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import unidecode


def get_statistics(data):
    """
        Calculate various statistics for a given dataset.

        Parameters:
        - data (list, numpy array, pandas DataFrame): The dataset for which statistics are calculated.
          If a pandas DataFrame is provided, statistics are calculated on the 'Count' column.

        Returns:
        - statistics (dict): A dictionary containing calculated statistics:
            - 'Mean': Mean value of the dataset.
            - 'Median': Median value of the dataset.
            - 'Standard Deviation': Standard deviation of the dataset.
            - 'Min': Minimum value in the dataset.
            - 'Max': Maximum value in the dataset.
            - 'Variance': Variance of the dataset.
            - 'Sum': Sum of all values in the dataset.
            - 'Number of Elements': Number of elements in the dataset.
            - 'Error' (if any): If an error occurs during calculation, it's captured and stored here.

        Notes:
        - If the input data is a pandas DataFrame, statistics are calculated on the 'Count' column.
        - Errors encountered during calculation are captured and stored in the 'Error' field of the dictionary.
    """
    statistics = {}
    try:
        if isinstance(data, pd.DataFrame):
            data = data['Count'].tolist()

        statistics['Średnia'] = np.mean(data)
        statistics['Mediana'] = np.median(data)
        statistics['Odchylenie standardowe'] = np.std(data)
        statistics['Min'] = np.min(data)
        statistics['Max'] = np.max(data)
        statistics['Wariancja'] = np.var(data)
        statistics['Suma'] = np.sum(data)
        statistics['Liczba elementów'] = len(data)
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        statistics['Error'] = str(e)
    return statistics
def normalize_name(name):
    """
        Normalize a given name by removing accents and converting it to lowercase.

        Parameters:
        - name (str): The name to be normalized.

        Returns:
        - str: The normalized name with accents removed and converted to lowercase.
    """
    return unidecode.unidecode(name).lower()


def calculate_language_numbers(df, language):
    """
        Calculate the total number of students learning a specific language across provinces.

        This function filters the DataFrame based on the specified language, normalizes the province names,
        and sums the student counts for each province.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing language and student count data.
        - language (str): The language for which the student counts are calculated.

        Returns:
        - pd.Series: A pandas Series with the total student counts per province for the specified language.
          Province names are normalized (lowercase and accents removed) for consistency.
    """
    df['Wojewodztwo'] = df['Wojewodztwo'].apply(normalize_name)
    language_counts = df[df['Język obcy'] == language].groupby('Wojewodztwo')['liczba uczniów'].sum()
    return language_counts.fillna(0)

def map_of_poland(language, year):
    """
        Generate a map of Poland showing student counts learning a specific language across provinces for a given year.

        This function reads the shapefile of Polish provinces, normalizes province names,
        calculates student counts for the specified language and year, merges this data with province shapes,
        and plots a choropleth map.

        Parameters:
        - language (str): The language for which student counts are displayed on the map.
        - year (int): The year for which student counts are displayed.

        Returns:
        - matplotlib.figure.Figure: A matplotlib figure object displaying the choropleth map.
    """
    voivodeships = gpd.read_file('data/wojewodztwa-max.geojson')
    voivodeships['nazwa'] = voivodeships['nazwa'].apply(normalize_name)

    if year == 2019:
        language_numbers = calculate_language_numbers(new_languages1, language)
    else:
        language_numbers = calculate_language_numbers(new_languages2, language)

    language_numbers = language_numbers.reset_index()

    voivodeships = voivodeships.merge(language_numbers, how='left', left_on='nazwa', right_on='Wojewodztwo')
    voivodeships['liczba uczniów'] = voivodeships['liczba uczniów'].fillna(0)

    cmap = plt.get_cmap('Purples')
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    voivodeships.plot(column='liczba uczniów', ax=ax, cmap=cmap, edgecolor='black', legend=True)
    plt.title(f'Mapa polskich wojewodztw - {language} ({year})')
    return fig


def num_of_students_plot():
    """
        Creates a pie chart showing the percentage distribution of foreign languages
        among students for the years 2019-2020 and 2023-2024.

        Uses data grouped from DataFrames new_languages1 and new_languages2,
        where 'Foreign Language' is the key and 'number of students' is the sum of students for each language.

        Returns:
        - fig (matplotlib.figure.Figure): A Matplotlib figure object containing the plot.

        Notes:
        - Each subplot represents data for one of the years (2019-2020 and 2023-2024).
        - Colors used in the plot are automatically generated for each dataset.
    """
    language_counts1 = new_languages1.groupby('Język obcy')['liczba uczniów'].sum()
    language_counts2 = new_languages2.groupby('Język obcy')['liczba uczniów'].sum()
    cmap1 = plt.get_cmap('Purples')
    cmap2 = plt.get_cmap('ocean')
    colors1 = cmap1(np.linspace(0.4, 0.9, len(language_counts1)))
    colors2 = cmap2(np.linspace(0.5, 0.9, len(language_counts2)))
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), squeeze=False)

    for i, (language_counts, colors, title) in enumerate(zip([language_counts1, language_counts2], [colors1, colors2],
                                                             ['Procent języków po ilości uczniów w latach(2019-2020)',
                                                              'Procent języków po ilości uczniów w latach(2023-2024)'
                                                              ])):
        ax = axs[0, i]
        ax.pie(language_counts, labels=language_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.set_title(title)
        ax.axis('equal')

    plt.tight_layout()
    return fig

def create_pivot_table(data, index_col, columns_col, values_col):
    """
       Create a pivot table from the given DataFrame.

       This function creates a pivot table where values are aggregated based on specified index, columns, and values.

       Parameters:
       - data (pd.DataFrame): The input DataFrame containing the data.
       - index_col (str or list of str): Column(s) to use as the index in the pivot table.
       - columns_col (str or list of str): Column(s) to use as the columns in the pivot table.
       - values_col (str): Column to aggregate values.

       Returns:
       - pd.DataFrame: A pivot table DataFrame where:
           - Rows represent unique values from the index columns.
           - Columns represent unique values from the columns columns.
           - Values are aggregated based on the specified values column using sum aggregation.
    """
    return data.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc='sum')
def plot_pivot_tables(pivot_table1, pivot_table2, title1, title2, xlabel, ylabel, figsize=(16, 8)):
    """
        Plot two side-by-side bar charts from two pivot tables.

        Parameters:
        - pivot_table1 (pd.DataFrame): First pivot table to plot.
        - pivot_table2 (pd.DataFrame): Second pivot table to plot.
        - title1 (str): Title for the first plot.
        - title2 (str): Title for the second plot.
        - xlabel (str): Label for the x-axis in both plots.
        - ylabel (str): Label for the y-axis in both plots.
        - figsize (tuple, optional): Figure size (width, height) in inches. Default is (16, 8).

        Returns:
        - matplotlib.figure.Figure: The matplotlib figure object containing the two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap1 = 'viridis'
    cmap2 = 'plasma'
    colors1 = plt.get_cmap(cmap1, len(pivot_table1.columns))
    colors2 = plt.get_cmap(cmap2, len(pivot_table2.columns))

    # Plotting the first pivot table
    for i, column in enumerate(pivot_table1.columns):
        pivot_table1[column].plot(kind='bar', ax=ax1, color=colors1(i / len(pivot_table1.columns)), label=column)
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(title='Język obcy')

    # Plotting the second pivot table
    for i, column in enumerate(pivot_table2.columns):
        pivot_table2[column].plot(kind='bar', ax=ax2, color=colors2(i / len(pivot_table2.columns)), label=column)
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(title='Język obcy')

    plt.tight_layout()
    return fig
def kategory_plot():
    """
        Generate and plot side-by-side bar charts of pivot tables for language students.

        Parameters:
        - new_languages1 (pd.DataFrame): DataFrame for years 2019-2020 containing language and student data.
        - new_languages2 (pd.DataFrame): DataFrame for years 2023-2024 containing language and student data.

        Returns:
        - matplotlib.figure.Figure: The matplotlib figure object containing the two side-by-side bar charts.
    """
    pivot_table1 = create_pivot_table(new_languages1, 'Kategoria uczniów', 'Język obcy', 'idKategoriaUczniow')
    pivot_table2 = create_pivot_table(new_languages2, 'Kategoria uczniów', 'Język obcy', 'idKategoriaUczniow')

    return plot_pivot_tables(pivot_table1, pivot_table2,
                             'Liczba uczniów według języka (2019-2020)',
                             'Liczba uczniów według języka (2023-2024)',
                             'Kategoria uczniów', 'Liczba uczniów')


def powiaty_po_wojewodztwam(wojewodzstwo):
    """
        Returns a list of counties ('powiaty') within a given voivodeship ('wojewodztwo')
        based on the data from new_languages1 DataFrame.

        Parameters:
        - wojewodztwo (str): Name of the voivodeship (province) to filter counties.

        Returns:
        - list: A list of counties ('powiaty') within the specified voivodeship.
    """
    miejscowosc_list = []

    for index, row in new_languages1.iterrows():
        if row['Wojewodztwo'] == wojewodzstwo:
            miejscowosc_list.append(row['Powiat'])
    return miejscowosc_list
def powiat_plot(wojewodzstwo):
    """
        Generate a scatter plot showing the distribution of foreign language learners across counties ('powiaty')
        within a specified voivodeship ('wojewodztwo') for two different years (2019/2020 and 2023/2024).

        Parameters:
        - wojewodztwo (str): Name of the voivodeship (province) for which county data will be plotted.

        Returns:
        - tuple: A tuple containing:
            - matplotlib.figure.Figure: The matplotlib figure object containing the scatter plots.
            - pd.DataFrame: DataFrame with language counts per county for 2019/2020.
            - pd.DataFrame: DataFrame with language counts per county for 2023/2024.
    """
    miejscowosc_list = powiaty_po_wojewodztwam(wojewodzstwo)
    powiat_data1 = new_languages1[new_languages1['Powiat'].isin(miejscowosc_list)]
    language_count1 = powiat_data1.groupby(['Powiat', 'Język obcy']).size().reset_index(name='Count')

    colors1 = {language: plt.cm.tab20(i / len(powiat_data1['Język obcy'].unique())) for i, language in
               enumerate(powiat_data1['Język obcy'].unique())}

    powiat_data2 = new_languages2[new_languages2['Powiat'].isin(miejscowosc_list)]
    language_count2 = powiat_data2.groupby(['Powiat', 'Język obcy']).size().reset_index(name='Count')

    colors2 = {language: plt.cm.tab20(i / len(powiat_data2['Język obcy'].unique())) for i, language in
               enumerate(powiat_data2['Język obcy'].unique())}

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].set_title('Języki obce w Powiacie 2019/2020')
    for key, group in language_count1.groupby('Język obcy'):
        if not group.empty:
            axs[0].scatter(group['Powiat'], group['Count'], s=100, color=colors1[key], label=key, alpha=0.6,
                           edgecolors='w')
    axs[0].set_xlabel('Powiat')
    axs[0].set_ylabel('Liczba języków')
    if language_count1['Język obcy'].nunique() > 0:
        axs[0].legend(title='Język obcy')
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].set_title('Języki obce w Powiacie 2023/2024')
    for key, group in language_count2.groupby('Język obcy'):
        if not group.empty:
            axs[1].scatter(group['Powiat'], group['Count'], s=100, color=colors2[key], label=key, alpha=0.6,
                           edgecolors='w')
    axs[1].set_xlabel('Powiat')
    axs[1].set_ylabel('Liczba języków')
    if language_count2['Język obcy'].nunique() > 0:
        axs[1].legend(title='Język obcy')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig, language_count1, language_count2
def podmiot_of_students_plot():
    """
        Generate a side-by-side bar plot showing the relationship between foreign language learning, type of entity ('Typ podmiotu'),
        and student counts for two different years (2019-2020 and 2023-2024).

        This function filters data for 'Typ podmiotu' ('Type of entity') from new_languages1 and new_languages2 DataFrames,
        creates pivot tables to aggregate student counts by 'Typ podmiotu' and 'Język obcy' ('Foreign language'),
        and then plots these pivot tables using plot_pivot_tables function.

        Returns:
        - matplotlib.figure.Figure: The matplotlib figure object containing the side-by-side bar plots.
    """
    podmiot1 = new_languages1[new_languages1['Typ podmiotu'].isin(podmioty)]
    podmiot2 = new_languages2[new_languages2['Typ podmiotu'].isin(podmioty)]

    pivot_table1 = create_pivot_table(podmiot1, 'Typ podmiotu', 'Język obcy', 'idTypPodmiotu')
    pivot_table2 = create_pivot_table(podmiot2, 'Typ podmiotu', 'Język obcy', 'idTypPodmiotu')

    return plot_pivot_tables(pivot_table1, pivot_table2,
                             'Diagram zależności języka a typu podmiota (2019-2020)',
                             'Diagram zależności języka a typu podmiota (2023-2024)',
                             'Typ podmiotu', 'Liczba uczniów')


def update_canvas(fig, canvas_frame, data1, data2):
    """
        Update the canvas with a new figure and display statistics for two datasets (data1 and data2).

        This function clears any existing widgets in the canvas_frame, calculates statistics using the get_statistics function,
        and displays these statistics along with the matplotlib figure on the canvas.

        Parameters:
        - fig (matplotlib.figure.Figure): The matplotlib figure object to be displayed on the canvas.
        - canvas_frame (tk.Frame): The tkinter frame where the figure and statistics will be displayed.
        - data1 (list, numpy array, pandas DataFrame or None): The data for which statistics will be displayed for year 2019/2020.
        - data2 (list, numpy array, pandas DataFrame or None): The data for which statistics will be displayed for year 2023/2024.

        Returns:
        - None
    """
    # Clear existing widgets in canvas_frame
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Create a frame for displaying statistics
    stats_frame = tk.Frame(canvas_frame)
    stats_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

    # Display statistics for data1 if provided
    if data1 is not None:
        stats1 = get_statistics(data1)
        stats_text1 = 'Statystyki lat 2019/2020:\n\n'
        for key, value in stats1.items():
            stats_text1 += f'{key}: {value:.2f}\n'
        statistics_label1 = tk.Label(stats_frame, text=stats_text1, justify='left', anchor='w', font=('Arial', 17))
        statistics_label1.pack(fill=tk.Y, expand=True)

    # Display statistics for data2 if provided
    if data2 is not None:
        stats2 = get_statistics(data2)
        stats_text2 = 'Statystyki lat 2023/2024:\n\n'
        for key, value in stats2.items():
            stats_text2 += f'{key}: {value:.2f}\n'
        statistics_label3 = tk.Label(stats_frame, text=stats_text2, justify='left', anchor='w',font= ('Arial', 17))
        statistics_label3.pack(fill=tk.Y, expand=True)

    # Embed the matplotlib figure on the canvas_frame using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
def on_button_click(item, canvas_frame):
    """
        Event handler for button click events in the GUI.

        Depending on the selected item (item), this function generates a specific plot,
        computes corresponding data for 2019/2020 and 2023/2024, and updates the canvas with
        the plot and statistics.

        Parameters:
        - item (str): The selected item from the GUI menu, determining which plot and data to use.
        - canvas_frame (tk.Frame): The tkinter frame where the plot and statistics will be displayed.

        Returns:
        - None
    """
    data1, data2 = None, None
    if item == "Zależności liczby uczniów i języków obcych":
            fig = num_of_students_plot()
            data1 = new_languages1.groupby('Język obcy')['liczba uczniów'].sum().dropna().values
            data2 = new_languages2.groupby('Język obcy')['liczba uczniów'].sum().dropna().values

    elif item == "Zależności kategorii uczniów i języków obcych":
            fig = kategory_plot()
            data1 = new_languages1.groupby('Kategoria uczniów')['idKategoriaUczniow'].sum().dropna().values
            data2 = new_languages2.groupby('Kategoria uczniów')['idKategoriaUczniow'].sum().dropna().values

    elif item == "Zależności typu podmiotu i języków obcych":
            fig = podmiot_of_students_plot()
            data1 = new_languages1.groupby('Typ podmiotu')['idTypPodmiotu'].sum().dropna().values
            data2 = new_languages2.groupby('Typ podmiotu')['idTypPodmiotu'].sum().dropna().values

    update_canvas(fig, canvas_frame, data1, data2)

def on_language_select(evt, canvas_frame):
    """
        Event handler for selecting a language from the listbox in the GUI.

        Retrieves the selected language, generates maps of Poland for years 2019 and 2023
        showing language distribution across provinces, and updates the canvas frames with
        the generated figures.

        Parameters:
        - evt (tk.Event): The event object triggered by selecting an item in the listbox.
        - canvas_frame (list of tk.Frame): List containing two tkinter frames where the figures
          will be displayed.

        Returns:
        - None
    """
    w = evt.widget
    if len(w.curselection()) == 0:
        return
    index = int(w.curselection()[0])
    language = w.get(index)

    # Generate maps for 2019 and 2023
    fig1 = map_of_poland(language, 2019)
    fig2 = map_of_poland(language, 2023)

    # Update canvas frames with the generated figures
    update_canvas(fig1, canvas_frame[0],None,None)
    update_canvas(fig2, canvas_frame[1],None,None)

def on_wojewodztwo_select(evt, canvas_frame):
    """
        Event handler for selecting a voivodeship (wojewodztwo) from the listbox in the GUI.

        Retrieves the selected voivodeship, generates a scatter plot showing foreign languages
        in counties (powiaty) for years 2019/2020 and 2023/2024, and updates the canvas frame with
        the generated figure and data.

        Parameters:
        - evt (tk.Event): The event object triggered by selecting an item in the listbox.
        - canvas_frame (tk.Frame): The tkinter frame where the figure will be displayed.

        Returns:
        - None
    """
    w = evt.widget
    if len(w.curselection()) == 0:
        return
    index = int(w.curselection()[0])
    wojewodztwo = w.get(index)
    fig, data1, data2 = powiat_plot(wojewodztwo)
    update_canvas(fig, canvas_frame, data1, data2)

def create_gui(new_langues1, new_langues2):
    """
        Creates a graphical user interface (GUI) with multiple tabs and interactive elements
        to display and interact with language learning data across different categories and regions.

        The GUI includes tabs for various diagrams and interactive selection elements for filtering data.

        Returns:
        - None
    """
    # Initialize the main Tkinter root window
    root = tk.Tk()
    root.title("Języki obce w latach 2019/2020 oraz 2023/2024")
    root.geometry("1400x1000")
    root.configure(background='white')

    # Create a main frame to hold all GUI components
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create a left pane for organizing GUI components
    left_pane = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
    left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a notebook widget to hold different tabs
    notebook = ttk.Notebook(left_pane)
    notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # List of items (tabs) to be displayed in the notebook
    items = [
        "Zależności liczby uczniów i języków obcych",
        "Zależności kategorii uczniów i języków obcych",
        "Zależności powiatów i języków obcych",
        "Zależności typu podmiotu i języków obcych",
        "Nauczenie języków obcych (województwa)"
    ]

    # Create each tab and add corresponding widgets
    for i, item in enumerate(items):
        tab_frame = ttk.Frame(notebook)
        tab_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        tab_frame.configure(width=500)

        notebook.add(tab_frame, text=item)

        if item == "Zależności powiatów i języków obcych":
                list_frame = ttk.Frame(tab_frame)
                list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
                diagram_frame = ttk.Frame(tab_frame)
                diagram_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

                wojewodzstwo_list = tk.Listbox(list_frame, height=10)
                wojewodzstwo_list.pack(fill=tk.X, expand=False)
                for wojewodztwo in wojewodztwa:
                    wojewodzstwo_list.insert(tk.END, wojewodztwo)
                wojewodzstwo_list.bind('<<ListboxSelect>>',lambda evt, df=diagram_frame: on_wojewodztwo_select(evt, df))

        elif item == "Nauczenie języków obcych (województwa)":
                list_frame = ttk.Frame(tab_frame)
                list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
                diagram_frame1 = ttk.Frame(tab_frame)
                diagram_frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                diagram_frame2 = ttk.Frame(tab_frame)
                diagram_frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

                languages_list = tk.Listbox(list_frame, height=10)
                languages_list.pack(fill=tk.X, expand=False)
                for language in ["angielski", "francuski", "niemiecki", "hiszpański", "włoski"]:
                    languages_list.insert(tk.END, language)

                languages_list.bind('<<ListboxSelect>>',
                                    lambda evt, df=[diagram_frame1, diagram_frame2]: on_language_select(evt, df))

        else:
                diagram_frame = ttk.Frame(tab_frame)
                diagram_frame.pack(fill=tk.BOTH, expand=True)
                button = ttk.Button(diagram_frame, text="Pokaż diagram",
                                    command=lambda i=item, df=diagram_frame: on_button_click(i, df))
                button.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()


if __name__ == '__main__':
    new_languages1 = pd.read_csv('data/languages2019.csv', low_memory=False)
    new_languages2 = pd.read_csv('data/languages2023.csv', low_memory=False)
    languages = ["angielski", "francuski", "niemiecki", "hiszpański", "włoski","rosyjski"]
    podmioty = ['Przedszkole', 'Branżowa szkoła I stopnia', 'Liceum ogólnokształcące', 'Liceum sztuk plastycznych',
                'Szkoła policealna', 'Technikum', 'Szkoła podstawowa', 'Punkt przedszkolny',
                'Branżowa szkoła II stopnia',
                'Ogólnokształcąca szkoła muzyczna I stopnia', 'Ogólnokształcąca szkoła muzyczna II stopnia']
    wojewodztwa=['DOLNOŚLĄSKIE', 'KUJAWSKO-POMORSKIE', 'LUBELSKIE', 'ŁÓDZKIE', 'MAŁOPOLSKIE', 'MAZOWIECKIE',
                                    'OPOLSKIE',
                                    'PODKARPACKIE', 'PODLASKIE', 'POMORSKIE', 'ŚLĄSKIE', 'ŚWIĘTOKRZYSKIE', 'WARMIŃSKO-MAZURSKIE',
                                    'WIELKOPOLSKIE', 'ZACHODNIOPOMORSKIE', 'LUBUSKIE']
    create_gui(new_languages1,new_languages2)