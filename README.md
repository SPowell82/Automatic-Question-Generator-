# Automatic-Question-Generator-
This program uses Natural Language Processing to automatically generate questions from web-based material. It produces the the best output with Wikipedia articles, but can be used with other websites. In order to function most optimally, these websites should contain a relatively large volume of text/information about the topic on the page with the provided link.

# Using the program in the terminal:

The current version is designed for users to operate the program in the terminal. If you only require the output of the program (i.e. the automatically generated questions), the program can be edited to generate questions for a single URL upon execution. This version is capable of storing many links (these can be search terms if Wikipedia is used, or URLs for other websites), which then require input from the terminal for the program to know which link or search term you want it to operate on. These search terms and URLs are stored in an array (navigate to the 'possibleTopics' function within the 'SelectWebsites' class). Therefore, if 'World War II' was used, the program would identify this as a Wikipedia search term as it is not a URL/link. This term can then be saved under the name history, so when the program requests what topic you would like questions for, you input 'history' in the terminal and it will generate questions from the Wikipedia article 'World War II'. Similarly, you may provide an array of many search terms and URLs related to historical articles. It will then randomly select a search term or link and generate questions for that page. This is good for quiz games. 

# Adding a link/searchterm:

Go to the 'SelectWebsites' class, and place a link to your desired website in an array inside the 'possibleTopics' function. If you are using Wikipedia, place the searchTerm only. Return this array and add a name for it in the terminal (i.e. history). The program will generate questions once you provide this name in the terminal.


# Updates:

There are several improvements to be made. For example, the process of randomly selecting a searchTerm of URL/link from an array is pseudorandom as it uses the python random libray. A better approach would be to use the 'secrets' libray:

-- from secrets import randbelow

This produces better random numbers. The current version also requires a relatively large volume of text to be on a single page. As a result, websites such as BBC bitesize will not produce questions of high quality as these websites spread the information related to one topic across multiple pages. Future improvements would identify when this is the case, and use all associated websites to produce the questions. 
