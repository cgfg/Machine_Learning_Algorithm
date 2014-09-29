The program is written in python, no make file will be need

The are five data set:
    voting
    monk1
    monk2
    car
    balance

Please use EXACTLY same name in command line

The first step is go to the folder "ID3"

The format of commands is
    python ./ID3.py [data_name] [check_mode] [run_id]

If you want to run new experiments, you don't need to provide run_id. If check_mode is true, it will load old experiments. If check_mode is false, it will run new experiments
run_id is provided in the report.

Example: run a new experiment on voting
    python ./ID3.py voting false

Example: run an old experiment whose run_id is 1
    python ./ID3.py voting true 1