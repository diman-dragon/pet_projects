import type { GrammarExerciseType } from "../types/domain";

export interface GrammarModule {
  title: string;
  language: string;
  level: string;
  section: string;
  tags: string;
  theory: string;
  examples: string;
  exercises: Array<{
    type: GrammarExerciseType;
    prompt: string;
    payload: string;
    answer: string;
    explanation: string;
  }>;
}

export const englishGrammarModules: GrammarModule[] = [
  {
    title: "Articles: A/An vs The",
    language: "English",
    level: "A1",
    section: "Articles",
    tags: "articles,basic",
    theory: `**Indefinite articles (A/An):**
- Use "a" before consonant sounds: a book, a cat, a university
- Use "an" before vowel sounds: an apple, an hour, an honest man
- First mention of something unknown to the listener

**Definite article (The):**
- Specific things known to both speaker and listener
- Unique objects: the sun, the moon, the president
- Second mention: "I saw a cat. The cat was black."

**No article:**
- General plurals: Cats are cute (not "The cats")
- Abstract nouns: Love is blind (not "The love")
- Meals, languages, days of week in general`,
    examples: `• I need a pen. (any pen)
• The pen is on the table. (specific pen)
• She is an engineer.
• An hour passes quickly.
• The Eiffel Tower is in Paris.
• Life is beautiful. (no article)`,
    exercises: [
      {
        type: "choice",
        prompt: "Choose the correct article: ___ apple a day keeps the doctor away.",
        payload: JSON.stringify({ options: ["A", "An", "The", "-"] }),
        answer: "An",
        explanation: "Use 'an' before words starting with a vowel sound (apple begins with the 'a' sound)."
      },
      {
        type: "choice",
        prompt: "___ sun rises in the east.",
        payload: JSON.stringify({ options: ["A", "An", "The", "-"] }),
        answer: "The",
        explanation: "There is only one sun in our solar system, so we use 'the' for unique objects."
      },
      {
        type: "fill_blank",
        prompt: "I saw ___ elephant at ___ zoo yesterday.",
        payload: "{}",
        answer: "an the",
        explanation: "'An elephant' - first mention, vowel sound. 'The zoo' - specific place both know."
      },
      {
        type: "fill_blank",
        prompt: "She is ___ honest person. ___ honesty is important.",
        payload: "{}",
        answer: "an -",
        explanation: "'Honest' starts with a vowel sound (silent 'h'). Abstract nouns don't need an article."
      },
      {
        type: "choice",
        prompt: "___ cats are independent animals.",
        payload: JSON.stringify({ options: ["A", "An", "The", "-"] }),
        answer: "-",
        explanation: "Plural nouns in general meaning don't need an article."
      },
      {
        type: "fill_blank",
        prompt: "I need ___ new car. ___ car I showed you is too expensive.",
        payload: "{}",
        answer: "a the",
        explanation: "'A' - any car. 'The' - specific car mentioned before."
      },
      {
        type: "reorder",
        prompt: "Build the phrase:",
        payload: JSON.stringify({ blocks: ["a", "university", "student"] }),
        answer: "a university student",
        explanation: "'University' starts with a consonant sound (y-university), so use 'a'."
      },
      {
        type: "translate",
        prompt: "Translate: Он купил книгу. Книга была интересная.",
        payload: "{}",
        answer: "He bought a book. The book was interesting.",
        explanation: "First mention - 'a book'. Second mention of same book - 'the book'."
      },
      {
        type: "choice",
        prompt: "___ President lives in ___ White House.",
        payload: JSON.stringify({ options: ["A/ a", "The/ the", "The/ -", "A/ the"] }),
        answer: "The/ the",
        explanation: "Both are unique: there is only one President (of the country) and one White House."
      },
      {
        type: "fill_blank",
        prompt: "I have ___ headache. Can you give me ___ aspirin?",
        payload: "{}",
        answer: "a an",
        explanation: "Fixed expressions: 'a headache', 'an aspirin' (aspirin starts with vowel sound)."
      },
      {
        type: "choice",
        prompt: "___ love makes ___ world go round.",
        payload: JSON.stringify({ options: ["The/ the", "-/ the", "The/ -", "-/ -"] }),
        answer: "-/ the",
        explanation: "Abstract nouns (love) don't need an article. 'The world' is unique."
      },
      {
        type: "reorder",
        prompt: "Build the phrase:",
        payload: JSON.stringify({ blocks: ["hour", "an", "need", "I"] }),
        answer: "I need an hour",
        explanation: "'Hour' starts with a vowel sound (silent 'h'), so use 'an'."
      }
    ]
  },
  {
    title: "Present Simple",
    language: "English",
    level: "A1",
    section: "Tenses",
    tags: "present,simple,tenses",
    theory: `**Form:**
Positive: I/You/We/They + V1 | He/She/It + V1+s/es
Negative: I/You/We/They don't + V1 | He/She/It doesn't + V1
Question: Do I/you/we/they + V1? | Does he/she/it + V1?

**Spelling 3rd person:**
- Most verbs: add -s (work → works)
- -s, -ss, -sh, -ch, -x, -o: add -es (go → goes)
- Consonant + y: -y + -ies (study → studies)
- Vowel + y: add -s (play → plays)

**Usage:**
- Habits, routines: I wake up at 7
- Facts, general truths: Water boils at 100°C
- Permanent situations: She lives in Paris
- Timetables: The train leaves at 9:00`,
    examples: `• I work from Monday to Friday.
• She doesn't like coffee.
• Do they speak Spanish?
• The shop opens at 9 AM.
• Water freezes at 0 degrees.
• He watches TV every evening.`,
    exercises: [
      {
        type: "choice",
        prompt: "She ___ to work by bus.",
        payload: JSON.stringify({ options: ["go", "goes", "going", "gone"] }),
        answer: "goes",
        explanation: "3rd person singular (she) needs -es after consonant + o: go → goes."
      },
      {
        type: "fill_blank",
        prompt: "I ___ (not/watch) TV very often.",
        payload: "{}",
        answer: "don't watch",
        explanation: "Negative in Present Simple: don't + verb (for I/you/we/they)."
      },
      {
        type: "choice",
        prompt: "___ they ___ football on Sundays?",
        payload: JSON.stringify({ options: ["Do/ play", "Does/ play", "Do/ plays", "Does/ plays"] }),
        answer: "Do/ play",
        explanation: "Questions with they: Do + subject + V1."
      },
      {
        type: "fill_blank",
        prompt: "The sun ___ (rise) in the east.",
        payload: "{}",
        answer: "rises",
        explanation: "General truth/fact uses Present Simple. 3rd person: rise → rises."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["usually", "drinks", "she", "tea"] }),
        answer: "she usually drinks tea",
        explanation: "Adverbs of frequency (usually) go before the main verb."
      },
      {
        type: "fill_blank",
        prompt: "Water ___ (boil) at 100 degrees Celsius.",
        payload: "{}",
        answer: "boils",
        explanation: "Scientific facts use Present Simple. 3rd person: boil → boils."
      },
      {
        type: "choice",
        prompt: "My brother ___ (study) medicine at university.",
        payload: JSON.stringify({ options: ["study", "studies", "studys", "studying"] }),
        answer: "studies",
        explanation: "Consonant + y changes to -ies: study → studies."
      },
      {
        type: "translate",
        prompt: "Translate: Они не говорят по-французски.",
        payload: "{}",
        answer: "They don't speak French.",
        explanation: "Negative with they: don't + V1."
      },
      {
        type: "fill_blank",
        prompt: "___ your father ___ (work) in an office?",
        payload: "{}",
        answer: "Does work",
        explanation: "Questions with he/she/it: Does + subject + V1."
      },
      {
        type: "choice",
        prompt: "I always ___ my teeth after breakfast.",
        payload: JSON.stringify({ options: ["brush", "brushs", "brushes", "brushing"] }),
        answer: "brush",
        explanation: "With I, use V1 without s. 'Always' shows it's a routine."
      },
      {
        type: "reorder",
        prompt: "Build the question:",
        payload: JSON.stringify({ blocks: ["live", "does", "London", "in", "she"] }),
        answer: "does she live in London",
        explanation: "Question form: Does + subject + V1 + ...?"
      },
      {
        type: "fill_blank",
        prompt: "He ___ (watch) the news every evening at 6 PM.",
        payload: "{}",
        answer: "watches",
        explanation: "Habit/routine. -ch ending adds -es: watch → watches."
      }
    ]
  },
  {
    title: "Past Simple: Regular Verbs",
    language: "English",
    level: "A1-A2",
    section: "Tenses",
    tags: "past,simple,regular",
    theory: `**Form:**
Positive: Subject + V2 (verb + -ed)
Negative: Subject + didn't + V1
Question: Did + subject + V1?

**Spelling rules for -ed:**
- Regular: work → worked
- -e: love → loved
- Consonant + y: study → studied
- Short vowel + consonant: stop → stopped
- Long vowel/consonant: open → opened

**Pronunciation:**
- /t/ after voiceless: watched, worked
- /d/ after voiced: played, opened
- /ɪd/ after t/d: wanted, needed

**Usage:**
- Completed actions in the past: I visited Paris last year.
- Past habits (not now): She smoked when she was young.
- Sequence of past events: He woke up, had breakfast, and left.`,
    examples: `• I walked to school yesterday.
• She didn't finish her homework.
• Did you see the movie?
• We lived in London in 2019.
• He stopped smoking last month.
• They planned a surprise party.`,
    exercises: [
      {
        type: "fill_blank",
        prompt: "I ___ (watch) a great movie last night.",
        payload: "{}",
        answer: "watched",
        explanation: "Past Simple for completed action. Regular: watch + ed."
      },
      {
        type: "choice",
        prompt: "She ___ (not/like) the food at the restaurant.",
        payload: JSON.stringify({ options: ["not liked", "didn't liked", "didn't like", "doesn't like"] }),
        answer: "didn't like",
        explanation: "Past Simple negative: didn't + V1 (no -ed)."
      },
      {
        type: "reorder",
        prompt: "Build the question:",
        payload: JSON.stringify({ blocks: ["did", "to", "the", "go", "you", "party"] }),
        answer: "did you go to the party",
        explanation: "Past Simple question: Did + subject + V1 + ...?"
      },
      {
        type: "fill_blank",
        prompt: "We ___ (live) in Paris for three years. (Now we live in Berlin)",
        payload: "{}",
        answer: "lived",
        explanation: "Past Simple for finished period in the past. Regular: live + d."
      },
      {
        type: "choice",
        prompt: "He ___ (study) medicine at university in the 1990s.",
        payload: JSON.stringify({ options: ["studied", "studyed", "studies", "study"] }),
        answer: "studied",
        explanation: "Consonant + y: change y to i and add -ed."
      },
      {
        type: "fill_blank",
        prompt: "___ they ___ (enjoy) the concert?",
        payload: "{}",
        answer: "Did enjoy",
        explanation: "Past Simple question: Did + subject + V1."
      },
      {
        type: "translate",
        prompt: "Translate: Я не закончил работу вчера.",
        payload: "{}",
        answer: "I didn't finish work yesterday.",
        explanation: "Past Simple negative: didn't + V1 (finish, not finished)."
      },
      {
        type: "choice",
        prompt: "The bus ___ (stop) at every station.",
        payload: JSON.stringify({ options: ["stoped", "stopped", "stopp", "stop"] }),
        answer: "stopped",
        explanation: "Short vowel + consonant: double the consonant before -ed."
      },
      {
        type: "fill_blank",
        prompt: "She ___ (plan) a trip to Italy last month.",
        payload: "{}",
        answer: "planned",
        explanation: "Short vowel + n: double n before -ed."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["ago", "arrived", "I", "an", "hour"] }),
        answer: "I arrived an hour ago",
        explanation: "Time expressions like 'an hour ago' go at the end."
      },
      {
        type: "choice",
        prompt: "They ___ (not/visit) the museum because it was closed.",
        payload: JSON.stringify({ options: ["not visited", "didn't visit", "didn't visited", "not visit"] }),
        answer: "didn't visit",
        explanation: "Past Simple negative: didn't + V1 (visit)."
      },
      {
        type: "fill_blank",
        prompt: "___ you ___ (watch) the match yesterday? Yes, I ___.",
        payload: "{}",
        answer: "Did watch did",
        explanation: "Question: Did + subject + V1. Short answer: Yes, I did."
      }
    ]
  },
  {
    title: "Prepositions of Place: In/On/At",
    language: "English",
    level: "A1",
    section: "Prepositions",
    tags: "prepositions,place",
    theory: `**In** - inside, enclosed spaces, large areas:
- in the room, in the box, in London, in England, in the world

**On** - surface, contact with surface:
- on the table, on the wall, on the floor, on the bus/train/plane (transport you can stand on)

**At** - exact point, specific location:
- at the door, at the bus stop, at home, at work, at school

**Special cases:**
- in the car / on the bus (enclosed vs standable)
- at the cinema (point) vs in the cinema (inside)
- on the corner (surface) vs at the corner (point)`,
    examples: `• The book is on the table.
• I live in Paris.
• She is at the door.
• The picture hangs on the wall.
• We are waiting at the bus stop.
• He is sitting in the car.`,
    exercises: [
      {
        type: "choice",
        prompt: "The cat is sleeping ___ the sofa.",
        payload: JSON.stringify({ options: ["in", "on", "at", "to"] }),
        answer: "on",
        explanation: "Surface contact = 'on'. The cat is on the surface of the sofa."
      },
      {
        type: "fill_blank",
        prompt: "I live ___ London, ___ England.",
        payload: "{}",
        answer: "in in",
        explanation: "Cities and countries use 'in' (enclosed/large area concept)."
      },
      {
        type: "choice",
        prompt: "She works ___ the hospital ___ the city center.",
        payload: JSON.stringify({ options: ["in/ at", "at/ in", "on/ at", "in/ on"] }),
        answer: "at/ in",
        explanation: "'At the hospital' (point), 'in the city center' (area)."
      },
      {
        type: "fill_blank",
        prompt: "The keys are ___ the table.",
        payload: "{}",
        answer: "on",
        explanation: "Objects resting on a surface use 'on'."
      },
      {
        type: "choice",
        prompt: "Wait for me ___ the bus stop.",
        payload: JSON.stringify({ options: ["in", "on", "at", "by"] }),
        answer: "at",
        explanation: "Specific point location = 'at'."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["in", "the", "box", "toys", "are", "the"] }),
        answer: "the toys are in the box",
        explanation: "Inside an enclosed space = 'in'."
      },
      {
        type: "fill_blank",
        prompt: "He is ___ work now. He will be ___ home at 6 PM.",
        payload: "{}",
        answer: "at at",
        explanation: "Fixed expressions: 'at work', 'at home' (specific locations)."
      },
      {
        type: "translate",
        prompt: "Translate: Картина висит на стене.",
        payload: "{}",
        answer: "The picture hangs on the wall.",
        explanation: "Surface = 'on'."
      },
      {
        type: "choice",
        prompt: "We are sitting ___ the classroom waiting ___ the teacher.",
        payload: JSON.stringify({ options: ["in/ for", "on/ at", "at/ for", "in/ at"] }),
        answer: "in/ for",
        explanation: "'In the classroom' (inside), 'waiting for' (fixed verb+prep)."
      },
      {
        type: "fill_blank",
        prompt: "There is a beautiful park ___ the corner of my street.",
        payload: "{}",
        answer: "at",
        explanation: "'At the corner' - specific point. 'On the corner' is also possible but less common."
      },
      {
        type: "choice",
        prompt: "I prefer to sit ___ the window when I travel by train.",
        payload: JSON.stringify({ options: ["in", "on", "at", "by"] }),
        answer: "by",
        explanation: "'By the window' = near/next to. Also common: 'at the window'."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["at", "I", "school", "was", "yesterday"] }),
        answer: "I was at school yesterday",
        explanation: "'At school' - specific institution point."
      }
    ]
  },
  {
    title: "Comparative and Superlative Adjectives",
    language: "English",
    level: "A2",
    section: "Adjectives",
    tags: "adjectives,comparison",
    theory: `**Comparative (-er/more):**
- Short (1-2 syllables): add -er (tall → taller, big → bigger)
- Long (2+ syllables): more + adj (beautiful → more beautiful)
- Irregular: good → better, bad → worse, far → farther/further

**Superlative (-est/most):**
- Short: add -est (tall → tallest)
- Long: most + adj (beautiful → most beautiful)
- Irregular: good → best, bad → worst

**Spelling rules:**
- y → i: happy → happier, happiest
- Short vowel: big → bigger, hot → hotter
- -e: large → larger, largest

**Usage:**
- Comparative: than (She is taller than me)
- Superlative: the + superlative (He is the tallest)

**Equality:**
- as + adj + as (as tall as, as beautiful as)`,
    examples: `• My house is bigger than yours.
• She is the most beautiful girl in the room.
• This exercise is easier than the last one.
• Today is hotter than yesterday.
• He runs as fast as his brother.
• This is the best movie I've ever seen.`,
    exercises: [
      {
        type: "choice",
        prompt: "She is ___ than her sister.",
        payload: JSON.stringify({ options: ["tall", "taller", "tallest", "more tall"] }),
        answer: "taller",
        explanation: "Short adjective + -er for comparison."
      },
      {
        type: "fill_blank",
        prompt: "This is the ___ (bad) day of my life.",
        payload: "{}",
        answer: "worst",
        explanation: "Irregular superlative: bad → worst."
      },
      {
        type: "choice",
        prompt: "My car is ___ (expensive) than yours.",
        payload: JSON.stringify({ options: ["expensiver", "more expensive", "most expensive", "expensiviest"] }),
        answer: "more expensive",
        explanation: "Long adjectives (3+ syllables) use 'more' for comparative."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["is", "as", "he", "tall", "as", "me"] }),
        answer: "he is as tall as me",
        explanation: "Equality comparison: as + adjective + as."
      },
      {
        type: "fill_blank",
        prompt: "This book is ___ (interesting) ___ that one.",
        payload: "{}",
        answer: "more interesting than",
        explanation: "Long adjective: more + adj + than."
      },
      {
        type: "choice",
        prompt: "Mount Everest is the ___ mountain in the world.",
        payload: JSON.stringify({ options: ["higher", "more high", "highest", "most high"] }),
        answer: "highest",
        explanation: "Short adjective + -est for superlative. The + superlative."
      },
      {
        type: "translate",
        prompt: "Translate: Это самый лучший день.",
        payload: "{}",
        answer: "This is the best day.",
        explanation: "Irregular superlative: good → best. The + superlative."
      },
      {
        type: "fill_blank",
        prompt: "She is ___ (happy) now because she got a promotion.",
        payload: "{}",
        answer: "happier",
        explanation: "y → i before -er: happy → happier."
      },
      {
        type: "choice",
        prompt: "The weather today is ___ than yesterday.",
        payload: JSON.stringify({ options: ["good", "better", "best", "more good"] }),
        answer: "better",
        explanation: "Irregular comparative: good → better."
      },
      {
        type: "reorder",
        prompt: "Build the sentence:",
        payload: JSON.stringify({ blocks: ["interesting", "most", "this", "is", "the", "book"] }),
        answer: "this is the most interesting book",
        explanation: "Superlative: the most + long adjective."
      },
      {
        type: "fill_blank",
        prompt: "I can't run ___ (fast) ___ you.",
        payload: "{}",
        answer: "as fast as",
        explanation: "Negative equality: not as + adj + as."
      },
      {
        type: "choice",
        prompt: "That was ___ (far) I've ever traveled.",
        payload: JSON.stringify({ options: ["farther", "the farthest", "further", "more far"] }),
        answer: "the farthest",
        explanation: "Superlative with 'the'. Far → farthest (distance)."
      }
    ]
  }
];

export default englishGrammarModules;
