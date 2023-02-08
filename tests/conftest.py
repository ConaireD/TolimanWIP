import pytest
import os
import shutil

rm: callable = lambda root: shutil.rmtree(root) if os.path.isdir(root) else pass 
mkdir: callable = lambda root: os.makedirs(root) if not os.path.isdir(root) else pass 

@pytest.fixture
def remove_installation(root: str) -> None:
    """
    """
    rm(root)
    yield
    rm(root)

@pytest.fixture
def make_fake_csv(
        root: str, 
        number_of_rows: int,
        number_of_columns: int,
    ) -> None:
    """
    """
    _: None = mkdir(root)
    file_name: str = "{}/fake.csv".format(root)
    with open(file_name, "w") as file:
        headers: str = ",".join(
            "header{}".format(column_header) 
            for column_header in range(number_of_columns)
        )
        file.write("{}{}".format(headers, os.linesep))
        
        for row in range(number_of_rows):
            line: str = ",".join(
                "{}".format(column) 
                for column in range(number_of_columns)
            ) + os.linesep
            file.write(line)
    return file_name            
    
