# code example 

    def expand_dataset_with_rotation(self, x_data, y_data, expand_times):
        """Expand dataset by applying random 3D rotations to sphere coordinates."""
        original_size = x_data.shape[0]
        total_size = original_size * (expand_times + 1)  # Original + expanded copies

        # Prepare arrays for expanded data
        expanded_x = torch.zeros((total_size, *x_data.shape[1:]), dtype=torch.float32)
        expanded_y = torch.zeros((total_size, *y_data.shape[1:]), dtype=torch.float32)

        # Copy original data
        expanded_x[:original_size] = x_data
        expanded_y[:original_size] = y_data

        # Generate rotated copies
        for i in range(expand_times):
            start_idx = (i + 1) * original_size

            # Generate random rotation matrix for this expansion
            rotation_matrix = self._generate_random_rotation_matrix()

            # Apply rotation to all events
            for j in range(original_size):
                # Rotate PMT positions (features 2, 3, 4 in each hit)
                x_rotated = self._rotate_x_data(x_data[j], rotation_matrix)

                # Rotate track positions (first 3 and last 3 coordinates)
                y_rotated = self._rotate_y_data(y_data[j], rotation_matrix)

                expanded_x[start_idx + j] = x_rotated
                expanded_y[start_idx + j] = y_rotated

        return expanded_x, expanded_y

    def _generate_random_rotation_matrix(self):
        """Generate a uniform random 3D rotation matrix using quaternions."""
        # Generate uniform random unit quaternion
        # Method: generate 3 random numbers in [0,1], use them to create uniform quaternion
        u1, u2, u3 = np.random.uniform(0, 1, 3)

        # Convert to uniform quaternion on 4D unit sphere
        q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

        # Convert quaternion to rotation matrix
        # q = [q0, q1, q2, q3] where q0 is scalar part
        q0, q1, q2, q3 = q0, q1, q2, q3

        # Rotation matrix from quaternion
        R = np.array(
            [
                [
                    1 - 2 * (q2**2 + q3**2),
                    2 * (q1 * q2 - q0 * q3),
                    2 * (q1 * q3 + q0 * q2),
                ],
                [
                    2 * (q1 * q2 + q0 * q3),
                    1 - 2 * (q1**2 + q3**2),
                    2 * (q2 * q3 - q0 * q1),
                ],
                [
                    2 * (q1 * q3 - q0 * q2),
                    2 * (q2 * q3 + q0 * q1),
                    1 - 2 * (q1**2 + q2**2),
                ],
            ]
        )

        return torch.tensor(R, dtype=torch.float32)

    def _rotate_x_data(self, x_event, rotation_matrix):
        """Rotate PMT positions in x_data event."""
        x_rotated = x_event.clone()

        # Find valid hits (non-zero positions)
        valid_hits = torch.any(x_event[:, 2:5] != 0, dim=1)

        if torch.any(valid_hits):
            # Extract positions (features 2, 3, 4)
            positions = x_event[valid_hits, 2:5]

            # Apply rotation
            rotated_positions = positions @ rotation_matrix.T

            # Update rotated positions
            x_rotated[valid_hits, 2:5] = rotated_positions

        return x_rotated

    def _rotate_y_data(self, y_event, rotation_matrix):
        """Rotate enter and exit positions in y_data event."""
        y_rotated = y_event.clone()

        # Extract enter position (first 3 coordinates) and exit position (last 3 coordinates)
        enter_pos = y_event[:3].reshape(1, 3)
        exit_pos = y_event[3:6].reshape(1, 3)

        # Apply rotation
        rotated_enter = enter_pos @ rotation_matrix.T
        rotated_exit = exit_pos @ rotation_matrix.T

        # Update rotated positions
        y_rotated[:3] = rotated_enter.flatten()
        y_rotated[3:6] = rotated_exit.flatten()

        return y_rotated
