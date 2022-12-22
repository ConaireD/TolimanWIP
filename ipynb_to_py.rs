fn main()
{
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2
    {
        panic!("Error: More than one argument recieved!");
    }

    let file_name: &String = &args[1];
 
    let ipynb: String = std::fs::read_to_string(file_name)
        .expect("File not found!");

	let _ipynb: File = std::io::File::open(file_name);
}
