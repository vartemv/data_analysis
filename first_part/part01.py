#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: [Your Name]

This project involves data processing and plotting using Python. 
This script includes functions for generating graphs, downloading and 
cleaning data, calculating distances, and timing execution.

Python libraries used:
- `requests`: For making HTTP requests to download data.
- `BeautifulSoup` (from `bs4`): For parsing HTML data.
- `numpy`: For efficient numerical computations.
- `matplotlib.pyplot`: For plotting graphs.
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any, Union
import time

url = 'https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html'

def decoratortimer(decimal: int) -> Callable:
    """
    Decorator function to time the execution of functions.

    Parameters:
    - decimal (int): Number of decimal places to display in the output.

    Returns:
    - A wrapper function that times the execution of a function and prints 
      the result in milliseconds.
    """
    def decoratorfunction(f: Callable) -> Callable:
        def wrap(*args, **kwargs) -> Any:
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print('{:s} function took {:.{}f} ms'.format(f.__name__, ((time2 - time1) * 1000.0), decimal))
            return result
        return wrap
    return decoratorfunction

def distance(a: np.array, b: np.array) -> np.array:
    """
    Calculates the Euclidean distance between two arrays of coordinates.

    Parameters:
    - a (np.array): Array of x and y coordinates.
    - b (np.array): Array of x and y coordinates.

    Returns:
    - np.array: Array of distances between corresponding points in `a` and `b`.
    """
    return np.sqrt((a[:, 0] - b[:, 0])**2 + (a[:, 1] - b[:, 1])**2)

def generate_graph(a: List[float], show_figure: bool = False, save_path: Union[str, None] = None) -> None:
    """
    Generates a graph of the sine function multiplied by squared values of `a`.

    Parameters:
    - a (List[float]): List of float values that define amplitude in the sine function.
    - show_figure (bool): Whether to display the plot.
    - save_path (Union[str, None]): Path to save the plot image. If None, the plot is not saved.

    Returns:
    - None
    """
    #Creating x and y values, and creating graph a**2 * sin(x) 
    x = np.linspace(0, 6 * np.pi, 1000)
    a_values = np.array(a)
    f_values = (a_values[:, np.newaxis] ** 2) * np.sin(x)

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for i, a in enumerate(a_values):
        plt.plot(x, f_values[i], label=fr'$y_{a}(x)$', linewidth=1.7)
        plt.fill_between(x, f_values[i], color=colors[i], alpha=0.1)

    #Formatting the plot
    plt.legend(loc='lower center', fontsize=12, ncol=len(a_values), bbox_to_anchor=(0.5, 1.02), borderaxespad=0.3, frameon=True)
    pi = np.pi
    plt.xticks([0, pi / 2, pi, 3 * pi / 2, 2 * pi, 5 * pi / 2, 3 * pi, 7 * pi / 2, 4 * pi, 9 * pi / 2, 5 * pi, 11 * pi / 2, 6 * pi],
               [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$', r'$\frac{5\pi}{2}$', r'$3\pi$', r'$\frac{7\pi}{2}$', r'$4\pi$', r'$\frac{9\pi}{2}$', r'$5\pi$', r'$\frac{11\pi}{2}$', r'$6\pi$'])
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(r'$f_a(x)$', fontsize=14)
    plt.xlim(0, 6 * np.pi)

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()

def generate_sinus(show_figure: bool = False, save_path: Union[str, None] = None) -> None:
    """
    Generates a plot of sinusoidal functions and displays regions in different colors.

    Parameters:
    - show_figure (bool): Whether to display the plot.
    - save_path (Union[str, None]): Path to save the plot image. If None, the plot is not saved.

    Returns:
    - None
    """
    #Creating x and y values for all graphs
    x = np.linspace(0, 100, 10000)
    x_ticks = np.arange(0, 101, 25)
    y_ticks = np.arange(-0.8, 0.81, 0.4)

    f_values1 = (0.5 * np.cos((np.pi * x) / 50))
    f_values2 = (0.25 * (np.sin(np.pi * x) + np.sin((3 * np.pi * x) / 2)))
    f_values3 = f_values1 + f_values2
    y_values = np.array([f_values1, f_values2, f_values3])

    fig, plots = plt.subplots(3, 1, figsize=(10, 6))

    #Dividing third graph into different parts for coloring purposes
    upper_part = np.ma.masked_where(y_values[0] > y_values[2], y_values[2])
    lower_part = np.ma.masked_where(y_values[0] < y_values[2], y_values[2])
    lower_part_right = np.ma.masked_where(x < 50, lower_part)
    lower_part_left = np.ma.masked_where(x > 50, lower_part)

    #Plotting all graphs
    for i, plts in enumerate(plots):
        if i == 2:
            plts.plot(x, lower_part_right, c='orange')
            plts.plot(x, lower_part_left, c='red')
            plts.plot(x, upper_part, c='green')
        else:
            plts.plot(x, y_values[i])
            plts.set_xticklabels([])
        plts.set_xlim(0, 100)
        plts.set_ylim(-0.8, 0.8)
        plts.set_xticks(x_ticks)
        plts.set_yticks(y_ticks)

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()

def clean_string(s: str) -> str:
    """
    Cleans a string by removing unwanted characters.

    Parameters:
    - s (str): String to be cleaned.

    Returns:
    - str: Cleaned string with specific characters removed and stripped of whitespace.
    """
    cleaned = s.replace('\xa0', '')
    cleaned = cleaned.replace('°', '')
    cleaned = cleaned.replace(',', '.')
    return cleaned.strip()

def download_data() -> Dict[str, List[Any]]:
    """
    Downloads geographical data from a URL and extracts relevant information.

    Returns:
    - dict: Dictionary containing lists of positions, latitudes, longitudes, and heights.
    """
    response = requests.get(url)

    data_dict = {
        'positions': [],
        'lats': [],
        'longs': [],
        'heights': []
    }

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        all_rows = soup.find_all('tr', class_='nezvyraznit')
        if all_rows:
            for index, rows in enumerate(all_rows):
                data = [td.get_text(strip=True) for td in rows.find_all('td')]
                cleaned_data = [clean_string(data[i]) for i in [0, 2, 4, 6]]
                data_dict['positions'].append(cleaned_data[0])
                data_dict['lats'].append(float(cleaned_data[1]))
                data_dict['longs'].append(float(cleaned_data[2]))
                data_dict['heights'].append(float(cleaned_data[3]))
    return data_dict

if __name__ == "__main__":
    # Test Euclidean distance calculation
    a = np.array([[0, 0], [0, 0], [2, 2]])
    b = np.array([[1, 1], [2, 2], [5, 6]])
    print("Euclidean distances:", distance(a, b))

    # Generate and show the graph
    print("Generating graph...")
    generate_graph([7, 4, 3], show_figure=True)

    # Generate and show the sinusoidal visualization
    print("Generating sinusoidal plot...")
    generate_sinus(show_figure=True)

    # Download and print meteorological station data
    print("Downloading meteorological station data...")
    data = download_data()
    print("Sample data:", {key: data[key][:2] for key in data})
