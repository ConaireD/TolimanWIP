use std::fs::File;
use std::io::Error;
use std::io::BufReader;

fn main()
{
    let args: Vec<String> = std::env::args().collect();
 
    if args.len() > 2
    {
        panic!("Error: More than one argument recieved!");
    }

    let file_name: &String = &args[1];
	let ipynbd = File::open(file_name);
	let buf_ipynb = BufReader::new(ipynb);

	for line in buf_ipynb.lines() 
	{
		println!(line);
	}
}
