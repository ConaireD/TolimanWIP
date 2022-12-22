fn main()
{
    let file_name: Vec<String> = std::env::args().collect();
 
    if file_name.len() > 1
    {
        panic!("Error: More than one argument recieved!");
    }
 
    let ipynb: String = std::fs::read_to_string(&file_name[0])
        .expect("File not found!");
}
