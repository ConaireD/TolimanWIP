use std::fs::File;
use std::io::{Write, BufReader, BufRead, Error};


// fn main() -> Result<(), Error> {
    // let path = "lines.txt";
// 
    // let mut output = File::create(path)?;
    // write!(output, "Rust is Fun")?;
// 
    // let input = File::open(path)?;
    // let buffered = BufReader::new(input);
// 
    // for line in buffered.lines() {
        // println!("{}", line?);
    // }
// 
    // Ok(())
// }


fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2 {
        panic!("Error: More than one argument recieved!");
    }

    let file_name = &args[1];
	let ipynb = File::open(file_name)?;
	let buf_ipynb = BufReader::new(ipynb);

	for line in buf_ipynb.lines() {
		println!("{}", line?);
	}

	return Ok(());
}
