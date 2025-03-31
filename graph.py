"""CSC111 Project 2: Smart Trades- Graph

Instructions (READ THIS FIRST!)
===============================

This Python module contains code to display the graph of various stock tickers.
The ticker's names are passed into the Smart Trades program (by the user).
Allows for the display of last year's stock data like stock price, RSI and MACD.

Copyright and Usage Information
===============================

This file is provided solely for the professional use of CSC111 adminstrators
at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult CSC111 Course Syllabus.

This file is Copyright (c) Smart Trades Team- Shaurya Sareen, Abish Kulkarni, Irin Jin
"""

import pandas as pd
import pygame
import matplotlib
import matplotlib.dates as mdates
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
matplotlib.use("Agg")


class DisplayGraph:

    """
    Allows for the display of last year's stock data like stock price, RSI and MACD, for valid stock tickers.

    Instance Attributes:
        - data: Contains a stock's (which was present in the dataset) stock price, RSI and MACD for last year
        - fig: A Matplotlib figure object used to display the stock graph
        - ax: A Matplotlib axis object that plots the graph, and sets the legend, x/y axis labels and the title

    Representation Invariants:
        - self.data is a Pandas DataFrame with a DateTimeIndex
        - self.data is non-empty (contains at least one row)
        - self.data contains relevant stock-related columns (e.g., 'RSI', 'MACD', etc.)
        - self.fig is a valid matplotlib figure object
        - self.ax is a valid matplotlib axis object
    """

    data: pd.DataFrame
    fig: plt.Figure
    ax: plt.Axes

    def __init__(self, data: pd.DataFrame, ticker: str) -> None:
        """
        Initializes a graph with filtered last year stock data for a specific ticker.
        Also initializes the graph's figure and axis.
        Calls the plot_graph() function to plot the data.
        
        Preconditions:
        - ticker is a valid stock's ticker, that is present in data
        - data is valid and contains the ticker's stock price, RSI and MACD for last year 
        """
        filtered_data = data[data.index.to_series().apply(lambda x: '2024-01-01' <= str(x) <= '2024-12-31')]
        self.data = filtered_data
        self.fig, self.ax = plt.subplots(figsize=[8, 4], dpi=110)
        self.plot_graph(ticker)

    def draw(self, screen: pygame.Surface, position: tuple[int, int]) -> None:
        """
        Draws the stock graph on the screen, at the specific position.

        Preconditions:
        - The position passed is within the screen width and height and can be seen by the user
        """
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        raw_data = canvas.get_renderer().buffer_rgba()
        surf = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
        screen.blit(surf, position)

    def plot_graph(self, ticker: str) -> None:
        """
        Plots the graph with the stock price, RSA and MACD, for the specific data provided for a ticker.
        Writes the title of the graph for the specific ticker.
        Shows the legend for the graph, and labels for the x and y axis.

        Preconditions:
        - ticker is a valid stock's ticker, that is present in self.data
        """
        self.ax.clear()
        self.ax.set_title(f"Stock Graph of {ticker.upper()}", fontsize=16, fontweight='bold', color='black')
        self.ax.plot(self.data.index, self.data['Close'], label="Stock Price", color="blue")
        self.ax.plot(self.data.index, self.data['RSI'], label="RSI", color="green", linestyle="dashed")
        self.ax.plot(self.data.index, self.data['MACD'], label="MACD", color="red", linestyle="dotted")
        self.ax.set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        self.ax.legend()
        self.ax.set_xlabel("Month of 2024")
        self.ax.set_ylabel("Price")
        self.fig.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=45, ha='right')
        plt.gcf().autofmt_xdate()

    def trace_line(self, screen: pygame.Surface, mouse_pos: tuple[int, int], graph_position: tuple[int, int]) -> None:
        """
        Draw a vertical line at the mouse position and display the data like RSI and MACD at that point in real-time.

        Preconditions:
        - mouse_pos accurately pin points the location of the user's mouse on the screen
        - graph_position is the position of the graph on the screen, which can be seen and is valid
        """
        x_pos = mouse_pos[0]

        self.data.index = pd.to_datetime(self.data.index).tz_localize(None)

        bottom_boundary = graph_position[1] + int(self.fig.get_size_inches()[1] * self.fig.dpi)

        x_pos = max(180, min(855, x_pos))

        date_at_mouse = self.data.index[0] + (self.data.index[-1] - self.data.index[0]) * ((x_pos - 180) / (855 - 180))

        closest_index = self.data.index.get_indexer([date_at_mouse], method='nearest')[0]

        row = self.data.iloc[closest_index]

        pygame.draw.line(screen, (255, 0, 0), (x_pos, graph_position[1]), (x_pos, bottom_boundary), 2)

        font = pygame.font.SysFont("Arial", 16)
        font.set_bold(True)
        text_lines = [
            f"Date: {self.data.index[closest_index].strftime('%Y-%m-%d')}",
            f"Stock Price: {float(row['Close'].item()):.2f}",
            f"RSI: {float(row['RSI'].item()):.2f}",
            f"MACD: {float(row['MACD'].item()):.2f}"
        ]

        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (x_pos + 10, graph_position[1] + 160 + i * 20))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['matplotlib', 'pygame', 'pandas'],
        'allowed-io': [],
        'max-line-length': 120,
        'disable': ['E9992', 'E9999']
    })
