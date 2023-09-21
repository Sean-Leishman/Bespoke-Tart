# Data Structure

`transcripts/` contains all conversations that occurs within the Maptask corpus.
It follow this format:
``` 
Found 2 args.
Observation: q5ec2; Result size: 194; atts: 2
who	$m	
f	uh-huh 	
g	right 	
g	go along to your left in a straight line about an inch on the map 	
f	what am i trying to avoid first of all 	
g	stony desert 	
f	that's below the start 	
g	uh-huh 	
f	so i just go s-- left 	
```

After parsing using the `transform_files` function the output is as follows:
```
uh-huh 
right go along to your left in a straight line about an inch on the map 
what am i trying to avoid first of all 
stony desert 
that's below the start 
uh-huh 
so i just go s-- left 
```
The lines alernate between speakers so each new line signifies a new speaker.
