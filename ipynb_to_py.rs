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
		let mut in_source_block: bool = false;
		let _line: &str = line.as_ref().unwrap();
		
		if _line.contains("\"source\": [") {
			println!("{}", _line);
			in_source_block = true;
		}

		if in_source_block {
			
		}

		if in_source_block && _line.ends_with("]") {
			println!("{}", _line);
		}
		
	}

	return Ok(());
}
