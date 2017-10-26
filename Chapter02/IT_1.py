import dit


# Suppose we have a really thick coin, one so thick that there is a reasonable chance of it landing on its edge. Here is how we might represent the coin in dit.
d = dit.Distribution(['H', 'T', 'E'], [.4, .4, .2])
print(d)

# Calculate the probability of H 
print(d['H'])

# Calculate the probability of the combination H or T.
print(d.event_probability(['H','T'])) 

# Calculate the Shannon entropy of the joint distribution.
entropy = dit.shannon.entropy(d)
print(entropy)

# Calculate the extropy of the joint distribution.
extropy = dit.other.extropy(d)
print(extropy)

import dit.example_dists
e = dit.example_dists.Xor()
e.set_rv_names(['X', 'Y', 'Z'])
print(e)

# Calculate the Shannon mutual informations I[X:Z]
xz = dit.shannon.mutual_information(e, ['X'], ['Z'])
print(xz)

#Calculate the Shannon mutual informations I[Y:Z]
yz = dit.shannon.mutual_information(e, ['Y'], ['Z'])
print(yz)

# Calculate the Shannon mutual informations I[X,Y:Z].
xyz = dit.shannon.mutual_information(e, ['X', 'Y'], ['Z'])
print(xyz)



