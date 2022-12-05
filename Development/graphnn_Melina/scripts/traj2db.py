import argparse
import os
import ase
import ase.db
import ase.io.trajectory


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert ASE trajectory file to ASE sqlite database",
    )
    parser.add_argument(
        "trajectory_files",
        type=str,
        nargs="*",
        help="Trajectory files to be concatenated to a database",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    if len(args.trajectory_files) > 1:
        db_filename = "merged_database.db"
    else:
        db_filename = os.path.splitext(args.trajectory_files[0])[0] + ".db"
    print("Writing to database file %s" % db_filename)
    num_atoms = 0
    with ase.db.connect(db_filename, append=False) as db:
        for traj_file in args.trajectory_files:
            print("Reading from %s" % traj_file)
            reader = ase.io.trajectory.TrajectoryReader(traj_file)
            for atoms in reader:
                db.write(atoms)
                num_atoms += 1
    print("Done, wrote %d atoms objects" % num_atoms)


if __name__ == "__main__":
    main()
