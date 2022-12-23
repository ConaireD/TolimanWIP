use std::fs::File;
use std::io::{BufReader, Error, BufRead, Write};


fn rename_ipynb_to_py(file_name: &String) -> String {
	return file_name
		.split(".")
		.collect::<Vec<&str>>()[0]
		.to_owned() + ".py";
}


fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2 {
        panic!("Error: More than one argument recieved!");
    }

    let file_name: &String = &args[1];
	let ipynb: File = File::open(file_name)?;
	let buf_ipynb: BufReader<File> = BufReader::new(ipynb);
	let mut in_source_block: bool = false;
	let mut py: File = File::create(rename_ipynb_to_py(file_name))?;

	for line in buf_ipynb.lines() {
		let _line: &str = line.as_ref().unwrap();
		
		if _line.ends_with("]") && in_source_block {
			in_source_block = false;
		}
	
		if in_source_block {
			let __line: &str = _line
				.trim()
				.trim_end_matches("\\n\",")
				.trim_start_matches("\"");
			write!(py, "{}\n", __line);
		}
			
		if _line.contains("\"source\": [") {
			in_source_block = true;
		}
	}

	return Ok(());
}