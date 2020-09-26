import numpy as np
import matplotlib.pyplot as plt
import random


def fetch_dataset(file_name, delimiter=','):
    house_size = np.array([])
    house_bedroom_count = np.array([])
    house_price = np.array([])
    file = open(file_name)
    for line in file:
        temp = line.split(delimiter)
        house_size = np.append(house_size, int(temp[0]))
        house_bedroom_count = np.append(
            house_bedroom_count, int(temp[1]))
        temp[2] = temp[2].replace('\n', '')
        house_price = np.append(house_price, int(temp[2]))
    file.close()
    return house_size, house_bedroom_count, house_price


def normalize_data(H_SIZE, H_BEDROOMCOUNT, H_PRICE, means, std_deviations):
    # subtract
    H_SIZE = np.subtract(H_SIZE, means[0])
    H_BEDROOMCOUNT = np.subtract(H_BEDROOMCOUNT, means[1])
    H_PRICE = np.subtract(H_PRICE, means[2])
    # divide
    H_SIZE = np.divide(H_SIZE, std_deviations[0])
    H_BEDROOMCOUNT = np.divide(H_BEDROOMCOUNT, std_deviations[1])
    H_PRICE = np.divide(H_PRICE, std_deviations[2])
    return H_SIZE, H_BEDROOMCOUNT, H_PRICE


def get_norms(H_SIZE, H_BEDROOMCOUNT, H_PRICE):
    means = []
    means.append(np.mean(H_SIZE))
    means.append(np.mean(H_BEDROOMCOUNT))
    means.append(np.mean(H_PRICE))
    std_divs = []
    std_divs.append(np.std(H_SIZE))
    std_divs.append(np.std(H_BEDROOMCOUNT))
    std_divs.append(np.std(H_PRICE))
    return means, std_divs


def main():

    # fetch dataset
    H_SIZE, H_BEDROOMCOUNT, H_PRICE = fetch_dataset('ex1data2.txt')

    # calculating normalization parameters
    means, std_divs = get_norms(H_SIZE, H_BEDROOMCOUNT, H_SIZE)

    # normalizing dataset
    H_SIZE, H_BEDROOMCOUNT, H_PRICE = normalize_data(
        H_SIZE, H_BEDROOMCOUNT, H_PRICE, means, std_divs)


if __name__ == "__main__":
    main()
