import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_name="sequences.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.initialize_database()

    def initialize_database(self):
        # self.cursor.execute("DROP TABLE Sort_Order")

        # Table for sequences if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS sequences (
            sequence_id INTEGER PRIMARY KEY,
            sequence_name TEXT NOT NULL,
            description TEXT,
            conditions TEXT,
            post_conditions TEXT,
            is_runnable_count INTEGER,
            is_runnable_condition TEXT,
            is_runnable_exit BOOLEAN
        );
        """)
        # Table for states if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS states (
            task_id INTEGER PRIMARY KEY,
            task_name TEXT NOT NULL,
            description TEXT,
            conditions TEXT,
            post_conditions TEXT,
            sequence_id INTEGER,
            FOREIGN KEY (sequence_id) REFERENCES sequences (sequence_id)
        );
        """)
        # Table for Operation_Sequence if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Operation_Sequence (
            sequence_id INTEGER PRIMARY KEY,
            operation_id INTEGER,
            sequence_name TEXT NOT NULL,
            object_name TEXT
        );
        """)
        # Table for Screw_Op_Parmeters if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Screw_Op_Parmeters (
            sequence_id INTEGER PRIMARY KEY,
            operation_order INTEGER, 
            object_id TEXT,
            rotation_dir BOOLEAN,
            number_of_rotations INTEGER,
            current_rotation INTEGER,
            operation_status BOOLEAN
        );
        """)
        # Table for Camera_Vision if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Camera_Vision (
            sequence_id INTEGER PRIMARY KEY,
            object_name TEXT,
            object_color TEXT,
            color_code TEXT,
            pos_X FLOAT,
            pos_Y FLOAT,
            pos_Z FLOAT,
            rot_X FLOAT,
            rot_Y FLOAT,
            rot_Z FLOAT,
            usd_name TEXT
        );
        """)
        # Table for Sort_Order if not already created
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Sort_Order (
            sequence_id INTEGER PRIMARY KEY,
            object_name TEXT,
            object_color TEXT
        );
        """)
        
        # Expected schema for the `sequences` table
        sequences_schema = [
            ("sequence_id", "INTEGER PRIMARY KEY"),
            ("sequence_name", "TEXT NOT NULL"),
            ("node_name", "TEXT"),
            ("description", "TEXT"),
            ("conditions", "TEXT"),
            ("post_conditions", "TEXT"),
            ("is_runnable_count", "INTEGER"), 
            ("is_runnable_condition", "TEXT"),
            ("is_runnable_exit", "BOOLEAN")
        ]
        self.update_table_schema("sequences", sequences_schema)

        # Expected schema for the `tasks` table
        state_schema = [
            ("task_id", "INTEGER PRIMARY KEY"),
            ("task_name", "TEXT NOT NULL"),
            ("description", "TEXT"),
            ("conditions", "TEXT"),
            ("post_conditions", "TEXT"),
            ("sequence_id", "INTEGER")
        ]
        self.update_table_schema("states", state_schema)

        # # Expected schema for the `Operation_Sequence` table
        Operation_Sequence_schema = [
            ("sequence_id", "INTEGER PRIMARY KEY"),
            ('operation_id', "INTEGER NOT NULL"),
            ("sequence_name", "TEXT NOT NULL"),
            ("object_name", "TEXT")
        ]
        self.update_table_schema("Operation_Sequence", Operation_Sequence_schema)

        # Expected schema for the `Screw_Op_Parmeters` table
        Screw_Op_Parmeters_schema = [
            ("sequence_id", "INTEGER PRIMARY KEY"),
            ('operation_order', "INTEGER NOT NULL"),
            ("object_id", "TEXT NOT NULL"),
            ("rotation_dir", "BOOLEAN NOT NULL"),
            ("number_of_rotations", "INTEGER NOT NULL"),
            ("current_rotation", "INTEGER NOT NULL"),
            ("operation_status", "BOOLEAN NOT NULL")
        ]
        self.update_table_schema("Screw_Op_Parmeters", Screw_Op_Parmeters_schema)

        # Expected schema for the `Camera_Vision` table
        Camera_Vision_schema = [
            ("sequence_id", "INTEGER PRIMARY KEY"),
            ("object_name", "TEXT NOT NULL"),
            ("object_color", "TEXT NOT NULL"),
            ("color_code", "TEXT"),
            ("pos_X", "FLOAT NOT NULL"),
            ("pos_Y", "FLOAT NOT NULL"),
            ("pos_Z", "FLOAT NOT NULL"),
            ("rot_X", "FLOAT NOT NULL"),
            ("rot_Y", "FLOAT NOT NULL"),
            ("rot_Z", "FLOAT NOT NULL"),
            ("usd_name", "TEXT NOT NULL")
        ]
        self.update_table_schema("Camera_Vision", Camera_Vision_schema)
        
        # Expected schema for the `Sort_Order` table
        Sort_Order_schema = [
            ("sequence_id", "INTEGER PRIMARY KEY"),
            ("object_name", "TEXT NOT NULL"),
            ("object_color", "TEXT NOT NULL")
        ]
        self.update_table_schema("Sort_Order", Sort_Order_schema)

        self.conn.commit()

    def update_table_schema(self, table_name, expected_columns):
        """
        Check and update the schema of a table to match the expected columns.

        Args:
            table_name (str): The name of the table to check.
            expected_columns (list of tuples): A list of tuples, where each tuple contains the column name and type.
                                            Example: [("sequence_id", "INTEGER"), ("sequence_name", "TEXT"), ...]

        Returns:
            None
        """
        # Get the current schema of the table
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {col[1]: col[2] for col in self.cursor.fetchall()}  # {column_name: column_type}

        # Compare with expected schema
        for column_name, column_type in expected_columns:
            if column_name not in existing_columns:
                # Add missing column to the table
                self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                print(f"Added column '{column_name}' to table '{table_name}'")
            # elif existing_columns[column_name].upper() != column_type.upper():
            #     print(f"Warning: Column '{column_name}' in table '{table_name}' has a mismatched type "
            #         f"(expected {column_type}, found {existing_columns[column_name]})")

        self.conn.commit()

    def populate_database(self):
        print("Database population started.")
        # Clearing the already existing tables
        self.cursor.execute("DELETE FROM sequences")
        self.cursor.execute("DELETE FROM states")
        self.cursor.execute("DELETE FROM Operation_Sequence")
        self.cursor.execute("DELETE FROM Screw_Op_Parmeters")
        self.cursor.execute("DELETE FROM Camera_Vision")
        self.cursor.execute("DELETE FROM Sort_Order")
        self.conn.commit()

        # Populate sequences
        sequences = [
        ("pick", "PickBlockRd", "Pick up an object", "gripper is clear", "object in gripper", 1, "aaa", 0),
        ("travel", "ReachToPlacementRd", "Move to the target location", "object in gripper", "at target location", 1, "aaa", 0),
        ("drop", "DropRd", "Drop the object", "at target location", "object dropped", 1, "aaa", 0),
        ("screw", "ScrewRd", "Screw the object two times", "task complete", "robot at home position", 1, "thresh_met and self.context.gripper_has_block", 1),
        ("go_home", "GoHome", "Return to the home position", "task complete", "robot at home position", 1, "aaa", 0),
        ]
        self.cursor.executemany(
            "INSERT INTO sequences (sequence_name, node_name, description, conditions, post_conditions, is_runnable_count, is_runnable_condition, is_runnable_exit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sequences
        )

        # Populate states
        states = [
        ("LiftState", "Lift an object vertically", "gripper is clear", "object in gripper", 1),
        ("SlideState", "Slide an object along X-axis", "object in gripper", "at target location", 1),
        ]
        self.cursor.executemany(
            "INSERT INTO states (task_name, description, conditions, post_conditions, sequence_id) VALUES (?, ?, ?, ?, ?)",
            states
        )

        # Populate Operation_Sequence
        Operation_Sequence = [( 1, "travel", ""),( 1, "pick", "BlueCube"),( 1, "travel", ""),( 1, "screw", "BlueCube"),( 1, "drop", ""),
                              ( 2, "travel", ""),( 2, "pick", "YellowCube"),( 2, "travel", ""),( 2, "screw", "YellowCube"),( 2, "drop", ""),
                              ( 3, "travel", ""),( 3, "pick", "GreenCube"),( 3, "travel", ""),( 3, "screw", "GreenCube"),( 3, "drop", ""),
                              ( 4, "travel", ""),( 4, "pick", "RedCube"),( 4, "travel", ""),( 4, "screw", "RedCube"),( 4, "drop", "")
        ]
        self.cursor.executemany(
            "INSERT INTO Operation_Sequence (operation_id, sequence_name, object_name) VALUES (?, ?, ?)",
            Operation_Sequence
        )

        # Populate Screw_Op_Parmeters
        # Fetch 'operation_order' values for 'screw' instances from Operation_Sequence
        self.cursor.execute("SELECT sequence_id, object_name FROM Operation_Sequence WHERE sequence_name = 'screw'")
        screw_data = self.cursor.fetchall()
        screw_op_parameters = [
            (i + 1, operation_order, object_name, i % 2 == 0, 3, 0, 0)  # Example values
            for i, (operation_order, object_name) in enumerate(screw_data)
        ]
        self.cursor.executemany(
            "INSERT INTO Screw_Op_Parmeters (sequence_id, operation_order, object_id, rotation_dir, number_of_rotations, current_rotation, operation_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            screw_op_parameters 
        )

        # Populate Camera_Vision
        Camera_Vision = [
        ("RedCube", "Red", "[0.7, 0.0, 0.0]",0.3, -0.4, 0.026, 0.0, 0.0, 0.0, "Cube.usd"),
        ("BlueCube", "Blue","[0.0, 0.0, 0.7]", 0.43, -0.4, 0.026, 0.0, 0.0, 0.0, "Cube.usd"),
        ("YellowCube", "Yellow","[0.7, 0.7, 0.0]", 0.56, -0.4, 0.026, 0.0, 0.0, 0.0, "Cube.usd"),
        ]
        self.cursor.executemany(
            "INSERT INTO Camera_Vision (object_name, object_color, color_code,pos_X, pos_Y, pos_Z, rot_X, rot_Y, rot_Z, usd_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            Camera_Vision
        )
        
        # Populate Sort_Order
        Sort_Order = [("RedCube", "Red"), ("BlueCube", "Blue"), ("YellowCube", "Yellow")]
        self.cursor.executemany(
            "INSERT INTO Sort_Order (object_name, object_color) VALUES (?, ?)",
            Sort_Order
        )
        self.conn.commit()
        print("Database populated successfully.")

    def query_task_info(self, table_name, condition_value, column_to_query, condition_column):
        if column_to_query:
            query = f"SELECT {column_to_query} FROM {table_name} WHERE {condition_column} = ?"
            self.cursor.execute(query, (condition_value,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        else:
            query = "SELECT * FROM tasks WHERE task_name = ?"
            self.cursor.execute(query, (condition_value,))
            row = self.cursor.fetchone()
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, row)) if row else None

    def update_table(self, table_name, column_to_update, new_value, condition_column, condition_value):
        """
        Update a specific column in any table with a new value based on a condition.

        Args:
            table_name (str): The name of the table to update.
            column_to_update (str): The column to update.
            new_value (any): The new value to set for the column.
            condition_column (str): The column to use for the condition.
            condition_value (any): The value to match in the condition column.
        """
        query = f"""
        UPDATE {table_name}
        SET {column_to_update} = ?
        WHERE {condition_column} = ?
        """
        self.cursor.execute(query, (new_value, condition_value))
        self.conn.commit()
        print(f"Updated {column_to_update} in {table_name} where {condition_column} = {condition_value} to {new_value}")

    def check_table_schema(self, table_name):
        """
        Print the schema of a table for debugging purposes.

        Args:
            table_name (str): The name of the table to check.
        """
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        schema = self.cursor.fetchall()
        print(f"Schema of {table_name}:")
        for column in schema:
            print(column)


    def close(self):
        self.conn.close()

if  __name__ == "__main__":
    db_path = os.path.join("/home/omniverse/.local/share/ov/pkg/isaac_sim-2023.1.1", "sequences.db")
    db_manager = DatabaseManager(db_path)
    db_manager.populate_database()
    db_manager.close()
