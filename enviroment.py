import torch
import numpy as np
from gym.spaces import Box


class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        # Define environment parameters
        self.size = (300, 300, 300)
        self.num_IoTD = 5  # Number of IoT devices
        self.start_position_A = np.array([0, 299, 0])  # UAV A start position
        self.start_position_B = np.array([299, 299, 200])  # UAV B start position
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)  # Age of Information counters
        self.energy_levels = np.ones(self.num_IoTD)  # Remaining energy of IoTDs (normalized)
        self.R_min = 0.1
        self.time = 0  # Time slot
        self.T = 150  # Total episode duration
        self.E = 0  # Total energy consumption
        self.A = 0  # Total AoI
        self.done = False

        # Define action space and observation space according to the paper
        # Action space: [dx_A, dy_A, dz_A, dx_B, dy_B, dz_B, rho, delta_1, delta_2, delta_3, delta_4, delta_5]
        self.action_space = Box(low=np.array([-1] * 6 + [0] + [0] * 5),
                                high=np.array([1] * 6 + [1] + [1] * 5))

        # State space: [x_A, y_A, z_A, x_B, y_B, z_B, AoI_1, ..., AoI_5, C_1, ..., C_5]
        self.observation_space = Box(low=np.array([0] * 16),
                                     high=np.array([300, 300, 300, 300, 300, 300] + [np.inf] * 5 + [1] * 5))

        # Define IoTD and eavesdropper positions
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ])
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ])

    def reset(self):
        # Reset environment to initial state
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)
        self.energy_levels = np.ones(self.num_IoTD)
        self.time = 0
        self.E = 0
        self.A = 0
        self.done = False

        # Construct state according to paper's definition
        state = np.concatenate((
            self.current_position_A,  # x_A, y_A, z_A
            self.current_position_B,  # x_B, y_B, z_B
            self.AoI,  # AoI for each IoTD
            self.energy_levels  # Remaining energy for each IoTD
        ))
        return state

    def energy(self, pos_A, pos_B):
        # Calculate propulsion energy consumption
        distance = np.linalg.norm(pos_A - pos_B)
        energy = (79.85 * (1 + 3 * (distance / 120) ** 2)
                  + 88.63 * np.sqrt(1 + 0.25 * (distance / 4.03) ** 4)
                  - 0.5 * np.sqrt((distance / 4.03) ** 2)
                  + 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * distance ** 3)
        return energy

    def step(self, action):
        # Parse action according to paper's definition
        # action: [dx_A, dy_A, dz_A, dx_B, dy_B, dz_B, rho, delta_1, ..., delta_5]
        dx_A, dy_A, dz_A = action[0], action[1], action[2]
        dx_B, dy_B, dz_B = action[3], action[4], action[5]
        rho = action[6]  # Data transmission time
        delta = action[7:]  # IoTD scheduling (one-hot encoded)

        # Initialize reward components
        r_A = 0  # AoI component
        r_E = 0  # Energy component
        r_P = 0  # Penalty component

        # Check if episode is done
        self.done = (self.time >= self.T)
        if self.done:
            state = self._get_state()
            return state, 0, self.done, {}

        # Calculate new positions with bounds checking
        new_A_x = int(self.current_position_A[0] + dx_A * 20)
        new_A_y = int(self.current_position_A[1] + dy_A * 20)
        new_A_z = int(self.current_position_A[2] + dz_A * 20)

        new_B_x = int(self.current_position_B[0] + dx_B * 20)
        new_B_y = int(self.current_position_B[1] + dy_B * 20)
        new_B_z = int(self.current_position_B[2] + dz_B * 20)

        # Handle boundary conditions with penalty
        out_of_bounds = False
        if (new_A_x < 0 or new_A_x > 299 or
                new_A_y < 0 or new_A_y > 299 or
                new_A_z < 0 or new_A_z > 299):
            new_A_x = np.clip(new_A_x, 0, 299)
            new_A_y = np.clip(new_A_y, 0, 299)
            new_A_z = np.clip(new_A_z, 0, 299)
            out_of_bounds = True

        if (new_B_x < 0 or new_B_x > 299 or
                new_B_y < 0 or new_B_y > 299 or
                new_B_z < 0 or new_B_z > 299):
            new_B_x = np.clip(new_B_x, 0, 299)
            new_B_y = np.clip(new_B_y, 0, 299)
            new_B_z = np.clip(new_B_z, 0, 299)
            out_of_bounds = True

        if out_of_bounds:
            r_P = -99  # Large penalty for going out of bounds

        # Update positions
        past_position_A = self.current_position_A
        past_position_B = self.current_position_B
        self.current_position_A = np.array([new_A_x, new_A_y, new_A_z])
        self.current_position_B = np.array([new_B_x, new_B_y, new_B_z])

        # Calculate energy consumption
        energy_A = self.energy(self.current_position_A, past_position_A)
        energy_B = self.energy(self.current_position_B, past_position_B)
        total_energy = energy_A + energy_B
        self.E += total_energy

        # Update AoI for all IoTDs
        self.AoI += 1

        # Calculate secure transmission rates
        R_sec = self._calculate_secure_rates()

        # Process selected IoTD (delta is one-hot encoded)
        selected_iotd = np.argmax(delta)

        # Check if secure transmission is possible for selected IoTD
        if R_sec[selected_iotd] > self.R_min and delta[selected_iotd] > 0.5:
            # Successful transmission - reset AoI and update reward
            r_A = -self.AoI[selected_iotd]  # Negative of AoI as in paper
            self.AoI[selected_iotd] = 0
            # Update energy level for the IoTD
            self.A += self.AoI[selected_iotd]
            self.energy_levels[selected_iotd] = max(0, self.energy_levels[selected_iotd] - 0.1)

        # Calculate total reward according to paper's formula: r_t = r^A(t) + r^E(t) + r^P(t)
        omega_P = 0.1  # Scaling factor to balance magnitude
        r_E = -omega_P * total_energy

        reward = r_A + r_E + r_P

        # Update time step
        self.time += 1

        # Get new state
        state = self._get_state()

        return state, reward, self.done, total_energy,self.A

    def _get_state(self):
        """Helper method to construct the state vector"""
        return np.concatenate((
            self.current_position_A,  # x_A, y_A, z_A
            self.current_position_B,  # x_B, y_B, z_B
            self.AoI,  # AoI for each IoTD
            self.energy_levels  # Remaining energy for each IoTD
        ))

    def _calculate_secure_rates(self):
        """Calculate secure transmission rates between UAV and IoTDs"""
        R_sec = np.zeros(self.num_IoTD)

        # Calculate UAV to IoTD distances
        Distance_UAV_IoTD = np.array([
            np.linalg.norm(self.current_position_A - pos)
            for pos in self.iotd_position
        ])

        # Calculate IoTD to eavesdropper distances
        Distance_IoTD_e = np.array([
            [np.linalg.norm(self.iotd_position[i] - self.e_position[j])
             for j in range(self.num_IoTD)]
            for i in range(self.num_IoTD)
        ])

        # Calculate UAV to Jammer (UAV B) distance
        Distance_UAV_Jammer = np.linalg.norm(self.current_position_A - self.current_position_B)

        # Calculate Jammer to eavesdropper distances
        Distance_Jammer_e = np.array([
            np.linalg.norm(self.current_position_B - pos)
            for pos in self.e_position
        ])

        # Calculate channel gains (simplified model)
        H_UAV_IoTD = np.sqrt(0.001 / (Distance_UAV_IoTD ** 2)) * (
                np.sqrt(1 / 2) * np.exp(-1j * 2 * np.pi * Distance_UAV_IoTD / 0.12) +
                np.sqrt(1 / 2) * np.random.normal(0, 1, self.num_IoTD)
        )

        H_IoTD_e = np.array([
            [np.sqrt(0.001 / (Distance_IoTD_e[i, j] ** 2)) * (
                    np.sqrt(1 / 2) * np.exp(-1j * 2 * np.pi * Distance_IoTD_e[i, j] / 0.12) +
                    np.sqrt(1 / 2) * np.random.normal(0, 1)
            ) for j in range(self.num_IoTD)]
            for i in range(self.num_IoTD)
        ])

        H_UAV_Jammer = np.sqrt(0.001 / (Distance_UAV_Jammer ** 2)) * (
                np.sqrt(1 / 2) * np.exp(-1j * 2 * np.pi * Distance_UAV_Jammer / 0.12) +
                np.sqrt(1 / 2) * np.random.normal(0, 1)
        )

        H_Jammer_e = np.sqrt(0.001 / (Distance_Jammer_e ** 2)) * (
                np.sqrt(1 / 2) * np.exp(-1j * 2 * np.pi * Distance_Jammer_e / 0.12) +
                np.sqrt(1 / 2) * np.random.normal(0, 1, self.num_IoTD)
        )

        R_D = np.zeros(5)
        R_E = np.zeros((5, 5))

        # Calculate rates
        for i in range(0, 5):
            R_D[i] = np.log2(1 + 0.01 * (H_UAV_IoTD[i] ** 2) / (1e-13 + 0.001 * (H_UAV_Jammer ** 2)))

        for x in range(0, 5):
            for y in range(0, 5):
                R_E[x][y] = np.log2(1 + 0.01 * (H_IoTD_e[x][y] ** 2) / (1e-13 + 0.001 * H_Jammer_e[y] ** 2))

        # Calculate secure rates
        for i in range(self.num_IoTD):
            R_sec[i] = max(0, R_D[i] - np.max(R_E[i, :]))

        return R_sec