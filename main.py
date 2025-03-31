"""CSC111 Project 2: Smart Trades- Main

Instructions (READ THIS FIRST!)
===============================

This Python module contains code to display the Smart Trades program UI.
The UI allows the user to type in stock tickers.
With this, calculations are done in the backend, using the calc module to determine whether the stock is a good investment.
It also displays a graph of the stock's last year data, using the graph module.
This module also contains code to run the graph module, with randomized data, as sample.

Copyright and Usage Information
===============================

This file is provided solely for the professional use of CSC111 adminstrators
at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult CSC111 Course Syllabus.

This file is Copyright (c) Smart Trades Team- Shaurya Sareen, Abish Kulkarni, Irin Jin
"""

import datetime
import pygame
import pandas as pd
import numpy as np
import yfinance as yf
from calc import Calculation
from graph import DisplayGraph


if __name__ == "__main__":

    #import python_ta
    #python_ta.check_all(config={
    #    'extra-imports': ['pandas', 'numpy', 'calc', 'graph', 'pygame', 'datetime', 'yfinance'],
    #    'allowed-io': [],
    #    'max-line-length': 140
    #})

    pygame.init()

    WIDTH, HEIGHT = 1000, 800
    WHITE = (240, 240, 240)
    BLACK = (30, 30, 30)
    GRAY = (180, 180, 180)
    BLUE = (70, 130, 180)
    GREEN = (34, 177, 76)
    RED = (200, 0, 0)
    FONT = pygame.font.Font(None, 36)
    TITLE_FONT = pygame.font.Font(None, 48)
    HINT_FONT = pygame.font.Font(None, 28)

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smart Trades")

    input_box = pygame.Rect(350, 150, 300, 50)
    button_box = pygame.Rect(375, 220, 250, 50)

    color_active = BLUE
    color_inactive = GRAY
    color = color_inactive
    active = False
    text = ""

    result_text = ""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1000)

    graph = None

    exception_reach = False

    show_sample = False  # IMPORTANT: Toggle this variable to True, to JUST show the sample graph (not the program UI or its functionality)
    sample_graph = None
    sample_data = None
    if show_sample:
        pygame.display.set_caption("Sample Graph")
        date_range = pd.date_range(start="2024-01-01", periods=365, freq="D")

        np.random.seed(42)
        close_prices = np.round(np.random.uniform(100, 200, size=365), 2)
        rsi_values = np.round(np.random.uniform(20, 80, size=365), 2)
        macd_values = np.round(np.random.uniform(-5, 5, size=365), 2)

        sample_data = pd.DataFrame({
            "Close": close_prices,
            "RSI": rsi_values,
            "MACD": macd_values
        }, index=date_range)
        sample_data.index = pd.to_datetime(sample_data.index)
        sample_data = sample_data.astype(float)
        sample_graph = DisplayGraph(sample_data, "Random Stock Ticker Data")

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not show_sample:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
                if button_box.collidepoint(event.pos):
                    calc_text = "Calculating..."
                    calc_surface = HINT_FONT.render(calc_text, True, GRAY)
                    text_x = (WIDTH - calc_surface.get_width()) // 2
                    text_y = input_box.y - 30
                    screen.blit(calc_surface, (text_x, text_y))
                    pygame.display.update()
                    try:
                        data = yf.download(text, start=start_date, end=end_date)
                        calcObj = Calculation(data)
                        prediction_data, model_data = calcObj.compute_data()
                        tree = calcObj.build_model(model_data)
                        result = calcObj.make_prediction(prediction_data, tree)
                        pred_copy = prediction_data.copy()
                        graph = DisplayGraph(pred_copy, text)

                        if result == 1:
                            exception_reach = False
                            result_text = f"Yes, invest in {text.upper()}"
                        else:
                            exception_reach = False
                            result_text = f"No, do not invest in {text.upper()}"
                    except ValueError:
                        exception_reach = True
                        result_text = "Sorry, the stock mentioned is not in our dataset"
                    text = ""
                    pygame.display.flip()
            elif event.type == pygame.KEYDOWN and not show_sample:
                if active:
                    if event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif len(text) < 20 and event.unicode.isalpha():
                        text += event.unicode

        if not show_sample:
            title_surface = TITLE_FONT.render("Smart Trades", True, BLACK)
            screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 50))

            line1 = "Enter a stock ticker [Examples: AAPL (Apple), NVDA (Nvidia),"
            line2 = "TSLA (Tesla), NTDOY (Nintendo), UBER (Uber))]:"

            text1_rect = HINT_FONT.render(line1, True, GRAY).get_rect(center=(WIDTH // 2, input_box.y - 40))
            text2_rect = HINT_FONT.render(line2, True, GRAY).get_rect(center=(WIDTH // 2, input_box.y - 20))

            screen.blit(HINT_FONT.render(line1, True, GRAY), text1_rect)
            screen.blit(HINT_FONT.render(line2, True, GRAY), text2_rect)

            pygame.draw.rect(screen, color, input_box, border_radius=10)
            txt_surface = FONT.render(text, True, BLACK)
            screen.blit(txt_surface, (input_box.x + 10, input_box.y + 10))

            pygame.draw.rect(screen, GREEN, button_box, border_radius=10)
            button_text = FONT.render("Predict", True, WHITE)
            screen.blit(button_text, (button_box.x + 80, button_box.y + 10))

            result_surface = FONT.render(result_text, True, GREEN if "Yes" in result_text else RED)
            if exception_reach:
                screen.blit(result_surface, (input_box.x - 100, input_box.y + 140))
            else:
                screen.blit(result_surface, (input_box.x + 30, input_box.y + 140))

            if graph and not exception_reach:
                graph.draw(screen, (70, 330))
                graph.trace_line(screen, pygame.mouse.get_pos(), (70, 330))
        else:
            sample_graph.draw(screen, (70, 150))
            sample_graph.trace_line(screen, pygame.mouse.get_pos(), (70, 150))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
