# Basic project starter code

def main():
    print("Project initialized successfully.")
    print("System running...")

    # Placeholder for sensor input
    sensor_value = 0

    # Simple simulation loop
    for i in range(5):
        sensor_value += i
        print(f"Sensor reading {i}: {sensor_value}")

    print("Execution complete.")

if __name__ == "__main__":
    main()
