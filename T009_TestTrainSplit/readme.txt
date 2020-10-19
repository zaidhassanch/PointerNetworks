sentenceDict has the following structure:

{
	wordArray: [
		{"word": "quick", "vector": [1,2,3,4]},
		{"word": "Brown", "vector": [1,2,3,4]},
		{"word": "Fox", "vector": [1,2,3,4]},
	],
	text : {}

}


1 - We will have a sentence
    ['The', 'quick', 'brown', 'fox']

2 - Append the index in it
    [['The', 0], ['quick', 1], ['brown', 2], ['fox', 3]]

3 - shuffle
    [ ['quick', 1], ['The', 0], ['fox', 3], ['brown', 2]]

4 - Append index again
    [ ['quick', 1, 0], ['The', 0, 1], ['fox', 3, 2], ['brown', 2, 3]]

5 - sort w.r.t. 2nd index (1)
    [ ['The', 0, 1], ['quick', 1, 0], ['brown', 2, 3], ['fox', 3, 2]]

6 - Get the target
   [1, 0, 3, 2]

7 - get embeddings from spacy
================================================================

8 - make batch for training, we have the text and the desired order

9 - "The quick brown dog" reordered on GPU

=================================================================
10 - CPU and GPU switching

11 - grouped sentences of same length together

12 - train the network on sentences of length 8

13 - Support different sentence length ranging from 4 to 12:

=================================================

14 - Generate Pickle file of data, tokenizing takes long
	> Prepare data in advance, took half a day,

=================================================

15 - 

15 - Save trained model






6:35 - 6:45   Planning
6:45 - 7:30    finished steps 1-7
7:30 - 8:00   break
8:00 - 8:50   running a sentence [A quick brown dog] different reordering and moving to GPU, worked
8:50 - 9:00   Planning for next 10, 11
9:05 - 9:40   break
9:40 - 9:50   figured out a way to make sentences of equal length, just append ". . ."
10:00 - 12:00 We were able to achieve so much

              CPU and GPU sw





# s1 = "He will address the nation today"
# t1 = nlp(s1)
# s2 = "What is your address"
# t2 = nlp(s2)

# t = t1[2].vector - t2[3].vector;
# print(t1[2].vector)
# print(t2[3].vector)
# print(t)
# exit()
