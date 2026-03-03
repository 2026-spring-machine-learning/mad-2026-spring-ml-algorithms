def calculate_bonus(sales):
    # sales = 20000
    bonus = sales * .1
    # Sales greater than 10000:
    if sales > 10000:
        bonus += sales * .05
    if sales > 30000:
        bonus += sales * .1

    return bonus


def main():
    name = 'Pablo'
    sales = 20000
    pablo_bonus = calculate_bonus(sales)
    # pablo_bonus = sales * .1
    # # Sales greater than 10000:
    # pablo_bonus += sales * .05

    name = 'Veronica'
    sales = 9500
    # veronica_bonus = sales * .1
    # Sales not greater than 10000:
    veronica_bonus = calculate_bonus(sales)

    name = 'Cindy'
    sales = 45000
    # cindy_bonus = sales * .1
    # # Sales greater than 10000:
    # cindy_bonus += sales * .05
    # # Sales greater than 30000:
    # cindy_bonus += sales * .1
    cindy_bonus = calculate_bonus(sales)

    # print(f'{name}\'s bonus is ${cindy_bonus:.2f}')
    print(f'Pablo\'s bonus is ${pablo_bonus:.2f}')
    print(f'Veronica\'s bonus is ${veronica_bonus:.2f}')
    print(f'{name}\'s bonus is ${cindy_bonus:.2f}')


if __name__ == "__main__":
    main()
