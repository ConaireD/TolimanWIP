use std::fs::File;
use std::io::Error;

fn main()
{
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2
    {
        panic!("Error: More than one argument recieved!");
    }

    let file_name: &String = &args[1];
	let ipynb: Result<File, Error> = File::open(file_name);
}
