# Word embeddings   

This project uses spacy and a directed community algorithm to 
parse a text and return corresponding communities of words.
Those communities are created thanks to the directed Louvain algorithm
developed by [Anthony Perez](https://www.univ-orleans.fr/lifo/membres/Anthony.Perez) and 
[Nicolas Dugué](https://lium.univ-lemans.fr/team/nicolas-dugue/).


<u>**Spacy:**</u>  
[Spacy](https://spacy.io/) is used to parse the text you give as an argument base on the 
[pipeline](https://spacy.io/usage/processing-pipelines) of your choice.

<u>**Directed louvain:**</u>    
To build communities of words we choose the use the [directed louvain](https://github.com/anthonimes/DirectedLouvain) algorithm
base on the [louvain](https://sites.google.com/site/findcommunities/) algorithm. 
This algorithm is based on the [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)) formula and will provide accurate
communities in a short execution time.
This project was built on the directed louvain to determine if the direction has an
impact on the output result.

---
## how to use
<u>**Dependancy:**</u>   
You will need to install spacy and the pipeline of your choice
to use this project.   
You can find more information on this link: [https://spacy.io/usage](https://spacy.io/usage)

<u>**Compilation:**</u>   
To use this project you should creat a folder (for instance `build`) inside the 
`DirectedLouvain/python-binding` folder.   
Inside this file you will compile the code to produce a python library,
to do so you should execute the command: 

    cmake ../python-binding

then:

    make

You should get a file with the extension .so .  
Drag this file into the root of the project and you should be all set.

<u>**Usage:**</u>      
You can use this project as stand-alone or as a module.   
In both case, you can set the path to the text you want to parse and 
the spacy pipeline to use.
