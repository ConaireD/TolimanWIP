import pytest
import time

def setup_then_remove(func: callable) -> None:
    print("Expensive operation ...")
    time.sleep(1)
    print("Done!")

    func(file)

    print("Cleaning up ...")
    time.sleep(1)
    print("Done!")
    
@pytest.mark.parametrize("file", ["a", "b", "c", "d"])
@setup_then_remove
def test_closure_time_baby(file) -> None:
    assert file.isalpha()



        
    
