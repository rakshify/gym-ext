"""Implements utities for the gridworld environment."""

import sys

from typing import List, Tuple


class GridRead:
    """Implements utities for the gridworld environment."""

    @staticmethod
    def _base_condition(tries: int, max_tries: int):
        """Base condition for all recursive functions."""
        if tries > max_tries:
            print(("You have reached the maximum number of tries,"
                   " please try again."))
            sys.exit(1)

    @classmethod
    def read_target_states(cls, row: int, col: int, tries: int,
                           max_tries: int = 3, read_float: bool = False
                           ) -> List[Tuple[int, int]]:
        """
        Reads target states for a grid.

        Args:
            row: Number of rows in grid.
            col: Number of columns in grid.
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.
            read_float: Whether to read float values.

        Returns:
            List of target states.
        """
        cls._base_condition(tries, max_tries)

        ip = input("Enter number of target states").strip()
        try:
            nt = int(ip)
            if nt <= 0:
                raise ValueError
        except ValueError:
            print("Number of target states has to be a positive integer")
            cls.read_target_states(tries + 1, max_tries)
        targets = []
        for _ in range(nt):
            state = cls.read_start_state(
                row, col, tries, max_tries, read_float)
            targets.append(state)
        return targets

    @classmethod
    def read_grid_size(cls, tries: int, max_tries: int = 3) -> Tuple[int, int]:
        """
        Reads grid size.

        Args:
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            Tuple of grid size.
        """
        cls._base_condition(tries, max_tries)

        text = "Enter the grid size, in space separated \"row column\" format."
        ip = input(text).strip().split()
        if len(ip) != 2:
            print("Currently only 2-D grids supported.")
            return cls.read_grid_size(tries + 1, max_tries)
        grid_size = []
        for i in ip:
            try:
                item = int(i.strip())
                if item <= 0:
                    print("Number of row/column must be greater than zero.")
                    return cls.read_grid_size(tries + 1, max_tries)
                grid_size.append(item)
            except ValueError:
                print("Grid row/column must be integers.")
                return cls.read_grid_size(tries + 1, max_tries)
        return tuple(grid_size)

    @classmethod
    def read_grid(cls, grid_size: Tuple[int, int], tries: int,
                  max_tries: int = 3) -> List[List[str]]:
        """
        Reads grid.

        Args:
            grid_size: Size of grid to read.
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            grid.
        """
        cls._base_condition(tries, max_tries)

        row, col = grid_size
        grid = []
        for i in range(row):
            text = input(f"Enter row-{i + 1} in space separated format:\t")
            text = text.strip().split()
            if len(text) != col:
                print(("Number of items provided in each row "
                       "should be equal to the number of columns."))
                cls.read_grid(grid_size, tries + 1, max_tries)
            grid_row = []
            for t in text:
                if t not in ("t", "o", "x"):
                    print("Grid cells can only have one of t/x/o as values.")
                    cls.read_grid(grid_size, tries + 1, max_tries)
                grid_row.append(t)
            grid.append(grid_row)
        return grid

    @classmethod
    def read_start_state(cls, row: int, col: int, tries: int,
                         max_tries: int = 3, read_float: bool = False
                         ) -> Tuple[int, int]:
        """
        Reads start state.

        Args:
            row: Number of rows in grid.
            col: Number of columns in grid.
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.
            read_float: Whether to read float values.

        Returns:
            Start state.
        """
        cls._base_condition(tries, max_tries)

        text = "Enter the start state, in space separated \"row col\" format."
        ip = input(text).strip().split()
        if len(ip) != 2:
            print("Currently only 2-D grids supported.")
            return cls.read_start_state(tries + 1, max_tries)
        idx = []
        j = 0
        for i in ip:
            j += 1
            try:
                item = float(i.strip()) if read_float else int(i.strip())
                if item < 0 or (j == 1 and item >= row) or \
                        (j == 2 and item >= col):
                    print(("State index should be valid from (0 0) to "
                           f"({row - 1} {col - 1}). Given ({ip[0]} {ip[1]})"))
                    return cls.read_start_state(row, col, tries + 1, max_tries)
                idx.append(item)
            except ValueError:
                print("Start state index must be integers.")
                return cls.read_start_state(row, col, tries + 1, max_tries)
        return tuple(idx)

    @classmethod
    def is_windy(cls, tries: int, max_tries: int = 3) -> bool:
        """
        Reads whether windy or not.

        Args:
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            Whether windy or not.
        """
        cls._base_condition(tries, max_tries)

        choice = input("Do you want the grid to be windy? (y/n):").lower()
        if choice not in ("y", "n", "yes", "no"):
            print("choice can only be either (y/n).")
            cls.is_windy(tries + 1, max_tries)
        return choice

    @classmethod
    def read_wind_direction(cls, tries: int, max_tries: int = 3) -> str:
        """
        Reads wind direction.

        Args:
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            Wind direction.
        """
        cls._base_condition(tries, max_tries)

        msg = "Enter wind direction (up(u), down(d), left(l), right(r)):"
        POSSIBLE_DIRECTIONS = (
            'up', 'down', 'left', 'right', 'u', 'd', 'l', 'r'
        )
        direction = input(msg).lower()
        if direction not in POSSIBLE_DIRECTIONS:
            print(f"Direction can only be one of {POSSIBLE_DIRECTIONS}")
            cls.read_wind_direction(tries + 1, max_tries)
        return direction

    @classmethod
    def read_wind_factors(cls, n: int, var: str, tries: int,
                          max_tries: int = 3) -> List[int]:
        """
        Reads wind factors.

        Args:
            n: Number of wind factors.
            var: Variable name.
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            Wind factors.
        """
        cls._base_condition(tries, max_tries)

        wind_factors = []
        for i in range(n):
            wf, tries = cls._read_single_wind_factor(var, i, tries, max_tries)
            wind_factors.append(wf)
        return wind_factors

    @classmethod
    def _read_single_wind_factor(cls, var: str, i: int, tries: int,
                                 max_tries: int = 3) -> Tuple[int, int]:
        """
        Reads single wind factor.

        Args:
            var: Variable name.
            i: Wind factor index.
            tries: Number of tries passed in reading.
            max_tries: Maximum number of tries to read.

        Returns:
            Wind factor.
        """
        cls._base_condition(tries, max_tries)

        wf = input(f"Enter wind-factor for {var}-{i + 1}:\t").strip()
        try:
            wf = int(wf.strip())
        except ValueError:
            print("Wind factor must be an integer.")
            cls._read_single_wind_factor(tries + 1, max_tries)
        return wf, tries
