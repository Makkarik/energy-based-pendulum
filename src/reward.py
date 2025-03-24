import numpy as np

ELEMENTS = 3
CHECK_VALUE = 11.772

class EnergyReward:
    """
    A class to calculate the energy-based reward for a reinforcement learning 
    environment.

    Parameters
    ----------
    pole1_length : float, optional
        The length of the first pole (default is 0.6).
    pole2_length : float, optional
        The length of the second pole (default is 0.6).
    cart_mass : float, optional
        The mass of the cart (default is 1).
    pole1_mass : float, optional
        The mass of the first pole (default is 1).
    pole2_mass : float, optional
        The mass of the second pole (default is 1).
    g : float, optional
        The gravitational acceleration (default is 9.81).

    Methods
    -------
    __call__(observation: np.ndarray) -> float
        Calculate the energy-based reward given the observation.
    """

    def __init__(
        self,
        pole1_length: float = 0.6,
        pole2_length: float = 0.6,
        cart_mass: float = 1,
        pole1_mass: float = 1,
        pole2_mass: float = 1,
        g: float = 9.81,
    ):
        """
        Initialize the reward by setting the environment's hyperparameters.

        Parameters
        ----------
        pole1_length : float, optional
            The length of the first pole (default is 0.6).
        pole2_length : float, optional
            The length of the second pole (default is 0.6).
        cart_mass : float, optional
            The mass of the cart (default is 1).
        pole1_mass : float, optional
            The mass of the first pole (default is 1).
        pole2_mass : float, optional
            The mass of the second pole (default is 1).
        g : float, optional
            The gravitational acceleration (default is 9.81).
        """
        self._cart_mass = cart_mass
        self._mass1 = pole1_mass
        self._mass2 = pole2_mass
        self._length1 = pole1_length
        self._length2 = pole2_length
        self._g = g

    def __call__(self, observation: np.ndarray) -> float:
        """
        Calculate the energy-based reward.

        Parameters
        ----------
        observation : np.ndarray
            The observation array containing the state of the system. It is expected to 
            have the following elements: [sin1, sin2, cos1, cos2, dx, dtheta1, dtheta2].
        Returns
        -------
        float
            The calculated energy-based reward.
        """
        sin1, sin2, cos1, cos2, dx, dtheta1, dtheta2 = observation[1:-1]

        e_t = np.zeros(shape=ELEMENTS)
        e_v = np.zeros(shape=ELEMENTS)

        # Calculate potential energies of the cart, first and second poles
        e_v[0] = 0
        e_v[1] = self._mass1 * self._g * self._length1 / 2 * cos1
        e_v[2] = self._mass2 * self._g * (
            self._length2 / 2 * cos2 + self._length1 * cos1
            )

        cos_diff = cos1 * cos2 + sin1 * sin2
        # Calculate the kinetic energies of the corresponding elements
        e_t[0] = self._cart_mass / 2 * dx**2
        e_t[1] = 7 / 24 * self._mass1 * self._length1**2 * dtheta1**2 
        e_t[2] = self._mass2 * self._length2 * dtheta2**2 / 6
        e_t[2] += self._mass2 * self._length2**2 / 2 * (
            dtheta1**2 + dtheta2**2 / 4 + dtheta1 * dtheta2 * cos_diff
            )

        return np.sum(e_v - e_t)


if __name__ == "__main__":
    # Sanity check
    reward = EnergyReward()
    best_obs = np.zeros(shape=9)
    best_obs[3:5] = 1
    max_reward = reward(best_obs)
    if max_reward - CHECK_VALUE < 1e-9:
        print(f"Maximum energy reward for default config is {max_reward:.3f}")
    else:
        msg = f"Unit test has been failed. Got {max_reward} instead of {CHECK_VALUE}!"
        raise AssertionError(msg)
