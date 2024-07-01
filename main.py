import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ema(period, data):
    result = data.copy()
    alpha = 2 / (period + 1)

    for i in range(len(data)):
        ema_value = 0

        if i < period:
            for j in range(i + 1):
                ema_value += data.iloc[i - j] * (1 - alpha) ** j
        else:
            for j in range(period):
                ema_value += data.iloc[i - j] * (1 - alpha) ** j

        result.iloc[i] = ema_value / sum((1 - alpha) ** k for k in range(min(i + 1, period)))

    return result


def calculations(data):
    # Określ okres dla EMA12 i EMA26

    # EMA12
    ema12 = ema(12, data['Close'])

    # EMA26
    ema26 = ema(26, data['Close'])

    macd = ema12 - ema26

    signal_line = ema(9, macd)

    return macd, signal_line


def calculate_buy_sell_points(macd, signal_line):
    buy = [0]
    sell = [0]

    for i in range(1, len(macd)):
        if macd.iloc[i] <= signal_line.iloc[i] and macd.iloc[i - 1] > signal_line.iloc[i - 1]:
            sell.append(1)
            buy.append(0)
        elif macd.iloc[i] >= signal_line.iloc[i] and macd.iloc[i - 1] < signal_line.iloc[i - 1]:
            buy.append(-1)
            sell.append(0)
        else:
            buy.append(0)
            sell.append(0)

    return buy, sell


def calculate_portfolio_values(data, macd, signal, starting_stock_amount=1000):
    buy, sell = calculate_buy_sell_points(macd, signal)

    portfolio_value = data['Close'].copy()

    initial_stock_amount = starting_stock_amount
    initial_money = 0
    portfolio_value.iloc[0] = initial_stock_amount * data['Close'].iloc[0]
    counter = 0
    transaction_impact = []

    print("Poczatkowa wartosc portfela: ", round(portfolio_value.iloc[0], 2), ",liczba akcji: ", initial_stock_amount,
          ",cena za jedna akcje: ", round(data['Close'].iloc[0], 2))

    for i in range(len(portfolio_value)):
        current_money = initial_money + initial_stock_amount * data['Close'].iloc[i]
        portfolio_value.iloc[i] = current_money

        if sell[i] == 1:
            initial_money += initial_stock_amount * data['Close'].iloc[i]
            initial_stock_amount = 0
            counter += 1
            sell_price = data['Close'].iloc[i]

            if i + 5 < len(data):
                sell_price_2 = data['Close'].iloc[i + 5]
                impact = (sell_price - sell_price_2) / sell_price * 100
                transaction_impact.append(impact)

        if buy[i] == -1:
            stock_bought = initial_money / data['Close'].iloc[i]
            initial_money -= stock_bought * data['Close'].iloc[i]
            initial_stock_amount += stock_bought
            counter += 1
            buy_price = data['Close'].iloc[i]

            if i + 5 < len(data):
                buy_price_2 = data['Close'].iloc[i + 5]
                impact = (buy_price - buy_price_2) / buy_price * 100
                transaction_impact.append(impact)

        if buy[i] == 0 and sell[i] == 0:
            transaction_impact.append(0)

    print("Portfel: ", round(portfolio_value.iloc[-1], 2), ",liczba akcji: ", round(initial_stock_amount, 2),
          ",cena za jedna akcje: ", round(data['Close'].iloc[-1], 2))
    print('\n')
    return portfolio_value, transaction_impact


def plot_calculations(MACD, SIGNAL_LINE, impact):
    cross_points_buy = [i for i in range(1, len(MACD))
                        if (MACD[i] > SIGNAL_LINE[i] and MACD[i - 1] <= SIGNAL_LINE[i - 1])]

    cross_points_sell = [i for i in range(1, len(MACD))
                         if (MACD[i] < SIGNAL_LINE[i] and MACD[i - 1] >= SIGNAL_LINE[i - 1])]

    cross_points_values_buy = [MACD[i] for i in cross_points_buy]
    cross_points_values_sell = [MACD[i] for i in cross_points_sell]

    non_zero_values = [x for x in impact if x != 0]
    mean_absolute_impact = np.mean(np.abs(non_zero_values))
    impact_1 = [impact[i] if np.abs(impact[i]) > mean_absolute_impact else 0
                for i in range(len(impact))]

    return cross_points_buy, cross_points_sell, cross_points_values_buy, cross_points_values_sell, impact_1


print('\n')

data_1000days = pd.read_csv("RACE.csv")[0:1000]

close_prices = data_1000days['Close'].tolist()
array = [close_prices[i + 1] - close_prices[i] for i in range(len(close_prices) - 1)]
mean_1000days = sum(array)/len(array)

print("Średnia wartość:", round(mean_1000days, 2))

data_last30days = data_1000days.iloc[970:1000].reset_index(drop=True)
data_first30days = data_1000days[0:30].reset_index(drop=True)

MACD_1000days, SIGNAL_LINE_1000days = calculations(data_1000days)
MACD_last30days, SIGNAL_LINE_last30days = calculations(data_last30days)
MACD_first30days, SIGNAL_LINE_first30days = calculations(data_first30days)

data_1000days['Date'] = pd.to_datetime(data_1000days['Date'])
data_1000days.set_index('Date', inplace=True)

plt.figure(figsize=(14, 7))
plt.title("Wykres notowań Ferrari N.V. (RACE)")
plt.plot(data_1000days.index, data_1000days['Close'], label='Cena zamknięcia', color='blue')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid()
plt.savefig('WykresNotowan.png')
plt.show()


portfolio_1000days, impact_1000days = calculate_portfolio_values(data_1000days, MACD_1000days, SIGNAL_LINE_1000days)
buy, sell = calculate_buy_sell_points(MACD_1000days, SIGNAL_LINE_1000days)

cross_points_buy, cross_points_sell, cross_points_values_buy, cross_points_values_sell, impact_1000days = \
    plot_calculations(MACD_1000days, SIGNAL_LINE_1000days, impact_1000days)

plt.figure(figsize=(14, 7))
plt.title("Wykres wskaźnika MACD i linii sygnałowej")
plt.plot(MACD_1000days, label='MACD')
plt.plot(SIGNAL_LINE_1000days, label='SIGNAL_LINE')
plt.scatter(cross_points_sell, cross_points_values_sell, color='red', marker='o', label='sell')
plt.scatter(cross_points_buy, cross_points_values_buy, color='green', marker='o', label='buy')

for i in range(1, len(impact_1000days)):
    if impact_1000days[i] != 0:
        if (sell[i] == 1):
            plt.annotate(f'{impact_1000days[i]: .2f}%',
                         (i, MACD_1000days[i]),
                         textcoords="offset points", xytext=(0, -15), ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow" if impact_1000days[i] < 0 else "blue", lw=1,
                                   alpha=0.7))
        if (buy[i] == -1):
            plt.annotate(f'{impact_1000days[i]: .2f}%',
                         (i, MACD_1000days[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="red" if impact_1000days[i] < 0 else "green", lw=1,
                                   alpha=0.7))
plt.xlabel('Dzień')
plt.ylabel('Wartość')
plt.legend()
plt.grid()
plt.savefig('MACD_SIGNAL.png')
plt.show()

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.title("Wykres wskaźnika MACD i linii sygnałowej")
plt.plot(MACD_1000days, label='MACD')
plt.plot(SIGNAL_LINE_1000days, label='SIGNAL_LINE')
plt.xlabel('Dzień')
plt.ylabel('Wartość')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title('Portfolio Value for the 1000 days')
plt.plot(data_1000days.index, portfolio_1000days, label='Portfolio value', color='blue')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('portfolio_1000.png')
plt.show()

portfolio_last30days, impact_last30days = calculate_portfolio_values(data_last30days, MACD_last30days,
                                                                     SIGNAL_LINE_last30days)
buy, sell = calculate_buy_sell_points(MACD_last30days, SIGNAL_LINE_last30days)

cross_points_buy, cross_points_sell, cross_points_values_buy, cross_points_values_sell, impact_last30days = \
    plot_calculations(MACD_last30days, SIGNAL_LINE_last30days, impact_last30days)

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.title("Wykres wskaźnika MACD i linii sygnałowej")
plt.plot(MACD_last30days, label='MACD')
plt.plot(SIGNAL_LINE_last30days, label='SIGNAL_LINE')
for i in range(1, len(impact_last30days)):
    if impact_last30days[i] != 0:
        if sell[i] == 1:
            plt.annotate(f'{impact_last30days[i]: .2f}%',
                         (i, MACD_last30days[i]),
                         textcoords="offset points", xytext=(0, -15), ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="red" if impact_1000days[i] < 0 else "green", lw=1,
                                   alpha=0.7))
        if buy[i] == -1:
            plt.annotate(f'{impact_last30days[i]: .2f}%',
                         (i, MACD_last30days[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="red" if impact_last30days[i] < 0 else "green", lw=1,
                                   alpha=0.7))
plt.xlabel('Dzień')
plt.ylabel('Wartość')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title('Portfolio Value for the 1000 days')
plt.plot(data_last30days.index, portfolio_last30days, label='Portfolio value', color='blue')
plt.annotate(f'{portfolio_last30days[0]: .2f}',
             (0, portfolio_last30days[0]), textcoords="offset points", xytext=(0, -15), ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", lw=1, alpha=0.7))

growth_percentage = portfolio_last30days[29] / portfolio_last30days[0] * 100

plt.annotate(f'{portfolio_last30days[29]: .2f} ({growth_percentage:.2f}%)',
             (29, portfolio_last30days[29]), textcoords="offset points", xytext=(0, -15), ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", lw=1, alpha=0.7))
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('portfolio_last30.png')
plt.show()

portfolio_first30days, impact_first30days = calculate_portfolio_values(data_first30days, MACD_first30days,
                                                                       SIGNAL_LINE_first30days)

cross_points_buy, cross_points_sell, cross_points_values_buy, cross_points_values_sell, impact_last30days = \
    plot_calculations(MACD_first30days, SIGNAL_LINE_first30days, impact_first30days)

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.title("Wykres wskaźnika MACD i linii sygnałowej")
plt.plot(data_first30days.index, MACD_first30days, label='MACD')
plt.plot(data_first30days.index, SIGNAL_LINE_first30days, label='SIGNAL_LINE')
plt.xlabel('Dzień')
plt.ylabel('Wartość')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title('Portfolio Value for the 30 days')
plt.plot(data_first30days.index, portfolio_first30days, label='Portfolio value', color='blue')
plt.annotate(f'{portfolio_first30days[0]: .2f}',
             (0, portfolio_first30days[0]),
             textcoords="offset points", xytext=(0, -15), ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", lw=1, alpha=0.7))
growth_percentage = portfolio_first30days[29] / portfolio_first30days[0] * 100
plt.annotate(f'{portfolio_first30days[29]: .2f} ({growth_percentage:.2f}%)',
             (29, portfolio_first30days[29]), textcoords="offset points", xytext=(0, -15), ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", lw=1, alpha=0.7))
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('portfolio_first30.png')
plt.show()

dates_to_check = [
    ('2022-03-18', 'Open', 'GP BAHRAIN 2022, 1-2 FERRARI'),
    ('2022-03-22', 'Open', 'GP BAHRAIN 2022, 1-2 FERRARI'),
    ('2023-09-15', 'Close', 'GP SINGAPOUR 2023, FERRARI WIN'),
    ('2023-09-20', 'Close', 'GP SINGAPOUR 2023, FERRARI WIN'),
    ('2024-01-31', 'Open', 'HAMILTON DO FERRARI'),
    ('2024-02-01', 'Close', 'HAMILTON DO FERRARI')
]

for target_date, column_name, race_name in dates_to_check:
    if target_date in data_1000days.index:
        value = data_1000days.loc[target_date, column_name]
        print(f"Wartość '{column_name}' dla daty ({race_name}) {target_date}: {value}")
    else:
        print(f"Brak danych dla daty {target_date}")
