# Mathematical Explanation of Generalization Improvement

## 1. The Seq2Seq Memorization Problem
Let $V$ be the vocabulary of characters. A standard Seq2Seq model attempts to learn the joint probability distribution of the split string $Y = (y_1, y_2, \dots, y_m)$ given the compound word $X = (x_1, x_2, \dots, x_n)$:
$$ P(Y|X) = \prod_{i=1}^m P(y_i | y_{<i}, X) $$
Because the model maps the entire sequence $X$ to $Y$, the space of possible mappings is $O(|V|^n \times |V|^m)$. Without structural constraints, the model tends to memorize exact $(X, Y)$ pairs seen during training, leading to poor generalization on out-of-vocabulary (OOV) compounds.

## 2. The Multi-Task Decompounding Approach
Our redesigned architecture decomposes $P(Y|X)$ into conditionally independent sub-tasks based on Paninian grammatical constraints:
1. **Boundary Detection**: $P(B|X)$, where $B \in \{0, 1\}^n$ indicates the locus of the phonetic change.
2. **Rule Classification**: $P(R|X)$, where $R \in \mathcal{R}$ is the finite set of Paninian phonetic rules (e.g., Dirgha Sandhi).
3. **Local Reconstruction**: $P(Y_{local} | B, R, X)$, generating only the necessary character transformations at the boundary.

$$ P(Y|X) \approx P(R|X) \cdot P(B|X) \cdot P(Y_{local} | B, R, X) $$

### Generalization Advantage
By predicting $R$ and $B$, the model learns the *rule* rather than the *word*. The number of Paninian rules $|\mathcal{R}|$ is very small (around 40 major valid Sandhi combinations). 
If the model correctly identifies the boundary $B$ (e.g., at the character 'ा') and the structural rule $R$ (e.g., $a + \bar{a} = \bar{a}$), it can flawlessly split an infinite number of novel compound words. Furthermore, by passing character representations through a **Phonetic Feature Encoder** that flags articulation place and vowel length, the attention mechanism learns phonetic similarity rules natively rather than having to blindly correlate independent character indices.

## 3. Constrained Decoding and Lexicon Validation
During inference, we maximize the constrained objective:
$$ \arg\max_{Y} \left( \log P(Y|X) + \alpha \log P_{lexicon}(Y) \right) $$
Where $P_{lexicon}(Y) = 1$ if the split components exist in the SQLite dictionary, and 0 otherwise.

If the neural confidence $\mathcal{C}(X) < \tau$ (hallucination detection), the system immediately falls back to the deterministic symbolic engine (Paninian rules directly coded), ensuring 100% validity for known phonetic junctions even if the neural model hallucinates.
