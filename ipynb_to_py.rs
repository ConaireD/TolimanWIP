use std::fs::File;
use std::io::{BufReader, Error, BufRead};

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2 {
        panic!("Error: More than one argument recieved!");
    }

    let file_name: &String = &args[1];
	let ipynb: File = File::open(file_name)?;
	let buf_ipynb: BufReader<File> = BufReader::new(ipynb);

	for line in buf_ipynb.lines() {
		match line {
			Ok(line_str) => println!("{}", line_str),
			Err(err) => println!("{}", err)
		}
		// if line.contains("soruce: [") {
			// println!("{}", line?);
		// }
	}

	return Ok(());
}
