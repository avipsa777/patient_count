import csv
from datetime import datetime, timedelta

def generate_data():
    data = []
    current_time = datetime.strptime("00:00:00", "%H:%M:%S")

    for i in range(1, 2001):
        arrival_time = current_time.strftime("%H:%M:%S")
        departure_time = (current_time + timedelta(minutes=5)).strftime("%H:%M:%S")

        data.append([i, arrival_time, departure_time])

        # Increment current time for the next entry
        current_time += timedelta(minutes=6)

    return data

def export_to_csv(data, filename="generated_times.csv"):
    with open(filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Sr No", "Arrival Time", "Departure Time"])
        csv_writer.writerows(data)

if __name__ == "__main__":
    generated_data = generate_data()
    export_to_csv(generated_data)
    print("Data exported to 'generated_times.csv'")
