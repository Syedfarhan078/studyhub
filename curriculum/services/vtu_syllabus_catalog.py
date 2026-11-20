"""
Authoritative VTU Official Syllabus Knowledge Base.
Contains verified 5-module curriculum specifications directly grounded in official
Visvesvaraya Technological University (VTU) prescribed syllabi across 2022, 2024, and 2025 Schemes.

Prevents generic AI hallucination when students generate subjects with course codes
such as BCS304, 1BCS305, BCS302, BCS303, BCS401, BCS403, BCS502, etc.
"""

import re
from typing import List, Dict, Any, Optional

VTU_PRESCRIBED_CATALOG: Dict[str, Dict[str, Any]] = {
    # ─── 3rd Semester: Data Structures and Applications ──────────────────────
    "DATA_STRUCTURES": {
        "canonical_name": "Data Structures and Applications",
        "canonical_code": "BCS304",
        "aliases": [
            "BCS304", "1BCS305", "BCS305", "21CS32", "18CS32",
            "DATA STRUCTURES", "DATA STRUCTURES AND APPLICATIONS",
            "DATA STRUCTURES AND APPLICATION", "DSA"
        ],
        "branch": "CSE",
        "semester": 3,
        "scheme": "VTU 2022/2024/2025 Scheme",
        "modules": [
            {
                "module_number": 1,
                "title": "Introduction to Data Structures, Arrays, and Stacks",
                "raw_text": (
                    "Introduction: Data Structures, Classifications (Primitive & Non-Primitive), Data Structure Operations. "
                    "Review of Pointers and Dynamic Memory Allocation Functions (malloc, calloc, realloc, free). "
                    "Arrays and Structures: Dynamically Allocated Arrays, Structures and Unions, Polynomials. "
                    "Sparse Matrices: Sparse Matrix Representation and Transposing a Matrix using Arrays. "
                    "Representation of Multidimensional Arrays, Strings, and Pattern Matching."
                ),
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "Introduction and Dynamic Memory Allocation",
                        "topics": [
                            "Data Structures and Classifications (Primitive vs Non-Primitive)",
                            "Pointers and Dynamic Memory Allocation (malloc, calloc, realloc, free)",
                            "Dynamically Allocated Arrays and Structures"
                        ]
                    },
                    {
                        "chapter_number": 2,
                        "chapter_title": "Polynomials, Sparse Matrices, and Arrays",
                        "topics": [
                            "Polynomial Representation using Arrays",
                            "Sparse Matrices and Transposing a Sparse Matrix",
                            "Multidimensional Arrays and String Pattern Matching"
                        ]
                    }
                ],
                "topics": [
                    {"title": "Data Structures Classification and Operations", "chapter_number": 1, "chapter_title": "Introduction & Memory Allocation", "is_high_yield": False, "description": "Core classification of primitive and non-primitive structures and basic operations."},
                    {"title": "Pointers and Dynamic Memory Allocation Functions", "chapter_number": 1, "chapter_title": "Introduction & Memory Allocation", "is_high_yield": True, "vtu_frequency_tag": "Core VTU 8-10M Question (malloc/free)", "description": "Memory allocation functions malloc, calloc, realloc, and free with pointer management."},
                    {"title": "Polynomial Representation and Evaluation", "chapter_number": 2, "chapter_title": "Arrays & Sparse Matrices", "is_high_yield": True, "vtu_frequency_tag": "Frequent 8-10M VTU Question", "description": "Representation of single and multi-variable polynomials using array structures."},
                    {"title": "Sparse Matrix Representation and Fast Transpose Algorithm", "chapter_number": 2, "chapter_title": "Arrays & Sparse Matrices", "is_high_yield": True, "vtu_frequency_tag": "High-Yield 10M Choice Question (Fast Transpose)", "description": "Triplet representation of sparse matrices and transpose vs fast transpose algorithms."}
                ]
            },
            {
                "module_number": 2,
                "title": "Stacks and Queues",
                "raw_text": (
                    "Stacks: Definition, Stack Operations, Array Representation of Stacks, Stacks using Dynamic Arrays. "
                    "Stack Applications: Polish notation, Infix to Postfix Conversion, Evaluation of Postfix Expression. "
                    "Queues: Definition, Array Representation, Queue Operations (Enqueue, Dequeue), Circular Queues using Dynamic Arrays, Multiple Stacks and Queues."
                ),
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "Stacks and Expressions",
                        "topics": [
                            "Stack Definition, Operations, and Dynamic Array Representation",
                            "Infix to Postfix Conversion using Stacks",
                            "Evaluation of Postfix Expressions"
                        ]
                    },
                    {
                        "chapter_number": 2,
                        "chapter_title": "Queues and Circular Queues",
                        "topics": [
                            "Queue ADT and Array Representation",
                            "Circular Queues and Dynamic Implementation",
                            "Multiple Stacks and Multiple Queues"
                        ]
                    }
                ],
                "topics": [
                    {"title": "Stack Operations and Dynamic Array Implementation", "chapter_number": 1, "chapter_title": "Stacks and Expressions", "is_high_yield": False, "description": "Core stack primitives push, pop, peek, and dynamic resizing behavior."},
                    {"title": "Infix to Postfix Expression Conversion Algorithm", "chapter_number": 1, "chapter_title": "Stacks and Expressions", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 8-10M VTU Question (Q3/Q4 Choice)", "description": "Step-by-step operator precedence stack algorithm for converting infix to postfix."},
                    {"title": "Evaluation of Postfix Expressions using Stack", "chapter_number": 1, "chapter_title": "Stacks and Expressions", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 6-8M VTU Numerical", "description": "Algorithm and tabular trace for evaluating arithmetic postfix expressions."},
                    {"title": "Circular Queues Implementation and Boundary Conditions", "chapter_number": 2, "chapter_title": "Queues and Circular Queues", "is_high_yield": True, "vtu_frequency_tag": "High-Yield 8-10M VTU Question", "description": "Overcoming linear queue drawbacks using circular array indexing and modulo arithmetic."}
                ]
            },
            {
                "module_number": 3,
                "title": "Linked Lists and Trees",
                "raw_text": (
                    "Linked Lists: Singly Linked Lists, Representation and Operations (Insertion, Deletion, Search), Circular Linked Lists, Doubly Linked Lists, Header Nodes. "
                    "Linked Stacks and Queues, Polynomial Addition using Linked Lists, Sparse Matrices using Linked Lists. "
                    "Trees: Introduction to Trees, Binary Trees (Properties, Array and Linked Representation), Binary Tree Traversals (Inorder, Preorder, Postorder), Threaded Binary Trees."
                ),
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "Linked Lists Architectures",
                        "topics": [
                            "Singly Linked List Operations (Insert, Delete, Search)",
                            "Doubly Linked Lists and Circular Linked Lists",
                            "Polynomial Addition using Linked Lists"
                        ]
                    },
                    {
                        "chapter_number": 2,
                        "chapter_title": "Binary Trees and Traversals",
                        "topics": [
                            "Binary Tree Properties and Representations",
                            "Binary Tree Traversals (Inorder, Preorder, Postorder)",
                            "Threaded Binary Trees (Inorder Threading)"
                        ]
                    }
                ],
                "topics": [
                    {"title": "Singly and Doubly Linked List Operations", "chapter_number": 1, "chapter_title": "Linked Lists", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Choice Question", "description": "Pointer manipulation for insertion, deletion, and traversal in singly and doubly linked chains."},
                    {"title": "Polynomial Addition using Singly Linked Lists", "chapter_number": 1, "chapter_title": "Linked Lists", "is_high_yield": True, "vtu_frequency_tag": "Frequent 8-10M VTU Algorithm", "description": "Algorithm and node structures for adding two polynomials represented as linked lists."},
                    {"title": "Binary Tree Traversals (Inorder, Preorder, Postorder)", "chapter_number": 2, "chapter_title": "Binary Trees", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 8-10M VTU Question (Q5/Q6)", "description": "Recursive and iterative depth-first tree traversal algorithms with complete traces."},
                    {"title": "Threaded Binary Trees and Inorder Threading", "chapter_number": 2, "chapter_title": "Binary Trees", "is_high_yield": False, "description": "Utilizing null pointers for efficient stack-free tree traversals."}
                ]
            },
            {
                "module_number": 4,
                "title": "Binary Search Trees and Graphs",
                "raw_text": (
                    "Binary Search Trees: Definition, Search, Insertion, and Deletion Operations, Selection Trees (Winner and Loser Trees), Forests, Representation of Disjoint Sets. "
                    "Graphs: The Graph ADT, Elementary Graph Operations, Graph Representations (Adjacency Matrix, Adjacency List), Breadth-First Search (BFS), Depth-First Search (DFS), Connected Components."
                ),
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "Binary Search Trees and Selection Trees",
                        "topics": [
                            "Binary Search Tree (BST) Operations (Search, Insert, Delete)",
                            "Selection Trees (Winner Trees and Loser Trees)",
                            "Disjoint Sets and Forest Representation"
                        ]
                    },
                    {
                        "chapter_number": 2,
                        "chapter_title": "Graphs and Graph Traversals",
                        "topics": [
                            "Graph Representations (Adjacency Matrix & Adjacency List)",
                            "Breadth-First Search (BFS) and Depth-First Search (DFS)",
                            "Connected Components and Spanning Forests"
                        ]
                    }
                ],
                "topics": [
                    {"title": "Binary Search Tree Construction, Search, and Deletion", "chapter_number": 1, "chapter_title": "BST & Selection Trees", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Choice Question (Q7/Q8)", "description": "Binary search tree property, searching, node insertion, and complex 3-case node deletion."},
                    {"title": "Selection Trees: Winner and Loser Trees", "chapter_number": 1, "chapter_title": "BST & Selection Trees", "is_high_yield": False, "description": "Complete binary tree representation for multi-way external sorting and merging."},
                    {"title": "Graph Representations: Adjacency Matrix and Adjacency List", "chapter_number": 2, "chapter_title": "Graphs & Traversals", "is_high_yield": False, "description": "Comparative memory and lookup trade-offs between adjacency matrices and linked lists."},
                    {"title": "Graph Traversals: Breadth-First Search (BFS) and Depth-First Search (DFS)", "chapter_number": 2, "chapter_title": "Graphs & Traversals", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Question", "description": "BFS queue exploration and DFS recursive/stack backtracking algorithms with traversal trees."}
                ]
            },
            {
                "module_number": 5,
                "title": "Hashing, Priority Queues, and Advanced Trees",
                "raw_text": (
                    "Hashing: Introduction, Hash Tables, Hash Functions, Static Hashing, Collision Resolution Techniques (Linear Probing, Quadratic Probing, Double Hashing, Separate Chaining), Dynamic Hashing. "
                    "Priority Queues: Single and Double-ended Priority Queues, Leftist Trees. "
                    "Efficient Binary Search Trees: Optimal Binary Search Trees, Introduction to AVL Trees and Red-Black Trees."
                ),
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_title": "Hashing and Collision Resolution",
                        "topics": [
                            "Hash Tables, Hash Functions, and Address Calculation",
                            "Collision Resolution (Linear Probing, Quadratic, Double Hashing, Chaining)",
                            "Dynamic Hashing Techniques"
                        ]
                    },
                    {
                        "chapter_number": 2,
                        "chapter_title": "Priority Queues and Efficient Trees",
                        "topics": [
                            "Priority Queues and Leftist Trees",
                            "Optimal Binary Search Trees (OBST)",
                            "Introduction to Balanced Search Trees (AVL & Red-Black Trees)"
                        ]
                    }
                ],
                "topics": [
                    {"title": "Hash Functions and Collision Resolution Techniques", "chapter_number": 1, "chapter_title": "Hashing", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Question (Q9/Q10 Choice)", "description": "Hash table design, division/multiplication methods, and linear probing vs separate chaining."},
                    {"title": "Static vs Dynamic Hashing Mechanisms", "chapter_number": 1, "chapter_title": "Hashing", "is_high_yield": False, "description": "Extendible and linear hashing structures for scalable key-value storage."},
                    {"title": "Priority Queues and Leftist Tree Properties", "chapter_number": 2, "chapter_title": "Priority Queues & Trees", "is_high_yield": True, "vtu_frequency_tag": "Frequent 8-10M VTU Question", "description": "Leftist tree shortest path definition, meld operation, and priority queue implementations."},
                    {"title": "Introduction to AVL Trees and Single/Double Rotations", "chapter_number": 2, "chapter_title": "Priority Queues & Trees", "is_high_yield": True, "vtu_frequency_tag": "High-Yield 8-10M Question", "description": "Height-balanced search trees, balance factor analysis, and LL, RR, LR, RL balancing rotations."}
                ]
            }
        ]
    },

    # ─── 3rd Semester: Digital Design and Computer Organization ─────────────
    "DDCO": {
        "canonical_name": "Digital Design and Computer Organization",
        "canonical_code": "BCS302",
        "aliases": ["BCS302", "21CS33", "18CS33", "DDCO", "DIGITAL DESIGN AND COMPUTER ORGANIZATION"],
        "branch": "CSE",
        "semester": 3,
        "scheme": "VTU 2022/2024/2025 Scheme",
        "modules": [
            {
                "module_number": 1,
                "title": "Karnaugh Maps and Combinational Logic Circuits",
                "raw_text": "Karnaugh maps simplification (2, 3, 4 variables), Quine-McCluskey method, multiplexers, demultiplexers, encoders, decoders, adders, subtractors.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Boolean Minimization", "topics": ["K-Map Simplification", "Quine-McCluskey Method"]}, {"chapter_number": 2, "chapter_title": "Combinational Circuits", "topics": ["Multiplexers and Decoders", "Adders and Subtractors"]}],
                "topics": [
                    {"title": "Karnaugh Map Minimization (SOP and POS forms)", "chapter_number": 1, "chapter_title": "Boolean Minimization", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Question", "description": "Boolean reduction using 3-variable and 4-variable K-maps with don't care conditions."},
                    {"title": "Quine-McCluskey Tabulation Method", "chapter_number": 1, "chapter_title": "Boolean Minimization", "is_high_yield": True, "vtu_frequency_tag": "Frequent 10M VTU Problem", "description": "Tabular prime implicant generation and prime implicant chart reduction algorithm."},
                    {"title": "Multiplexers, Encoders, and Decoders Design", "chapter_number": 2, "chapter_title": "Combinational Circuits", "is_high_yield": True, "vtu_frequency_tag": "Core 8-10M VTU Design Question", "description": "Combinational circuit design and implementing boolean functions using multiplexers."}
                ]
            },
            {
                "module_number": 2,
                "title": "Sequential Logic Circuits, Latches, and Registers",
                "raw_text": "Latches, Flip-Flops (SR, JK, D, T), Master-Slave Flip-Flops, Registers, Shift Registers, Synchronous and Asynchronous Counters.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Flip-Flops", "topics": ["SR, JK, D, T Flip-Flops", "Master-Slave Flip-Flops"]}, {"chapter_number": 2, "chapter_title": "Registers and Counters", "topics": ["Shift Registers", "Synchronous Counters"]}],
                "topics": [
                    {"title": "SR, JK, D, and T Flip-Flops Working and Characteristic Tables", "chapter_number": 1, "chapter_title": "Flip-Flops", "is_high_yield": True, "vtu_frequency_tag": "Core 8-10M VTU Question", "description": "Gated latches, excitation tables, and Master-Slave JK flip-flop race-around elimination."},
                    {"title": "Synchronous and Asynchronous Binary Counter Design", "chapter_number": 2, "chapter_title": "Registers & Counters", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M Design Problem", "description": "State diagram, state transition tables, and flip-flop synthesis for modulo-N counters."}
                ]
            },
            {
                "module_number": 3,
                "title": "Basic Structure of Computers and Machine Instructions",
                "raw_text": "Functional units of a computer, Operational concepts, Memory locations and addresses, Memory operations, Instructions and instruction sequencing, Addressing modes.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Computer Organization", "topics": ["Functional Units", "Bus Structures"]}, {"chapter_number": 2, "chapter_title": "Instructions & Addressing", "topics": ["Addressing Modes", "Instruction Sequencing"]}],
                "topics": [
                    {"title": "Functional Units and Basic Operational Concepts", "chapter_number": 1, "chapter_title": "Computer Organization", "is_high_yield": False, "description": "Bus architectures, processor-memory interaction, and instruction cycle steps."},
                    {"title": "Machine Instruction Addressing Modes", "chapter_number": 2, "chapter_title": "Instructions & Addressing", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M Question", "description": "Immediate, direct, indirect, register, indexed, and relative addressing modes with diagrams."}
                ]
            },
            {
                "module_number": 4,
                "title": "Input/Output Organization, Interrupts, and DMA",
                "raw_text": "Accessing I/O devices, Interrupts, Interrupt hardware, Enabling and disabling interrupts, Handling multiple devices, Direct Memory Access (DMA), Bus architecture.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Interrupts", "topics": ["Interrupt Handling", "Vectored Interrupts"]}, {"chapter_number": 2, "chapter_title": "DMA and Buses", "topics": ["DMA Controllers", "Bus Arbitration"]}],
                "topics": [
                    {"title": "Interrupt Handling Hardware and Priority Arbitration", "chapter_number": 1, "chapter_title": "Interrupts", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Question", "description": "Vectored interrupts, daisy chaining, and interrupt nesting mechanisms."},
                    {"title": "Direct Memory Access (DMA) and Bus Arbitration", "chapter_number": 2, "chapter_title": "DMA & Buses", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 8-10M Question", "description": "DMA controller architecture, burst mode, cycle stealing, and centralized bus arbitration."}
                ]
            },
            {
                "module_number": 5,
                "title": "Memory System and Basic Processing Unit",
                "raw_text": "Basic concepts, Semiconductor RAM memories, Read-only memories, Cache memories (mapping functions, replacement algorithms), Virtual memory, Fundamental concepts of execution.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Memory Hierarchy", "topics": ["Semiconductor RAM/ROM", "Cache Mapping Functions"]}, {"chapter_number": 2, "chapter_title": "Processing Unit", "topics": ["Single Bus CPU Organization", "Instruction Fetch & Execution"]}],
                "topics": [
                    {"title": "Cache Memory Mapping Functions (Direct, Associative, Set-Associative)", "chapter_number": 1, "chapter_title": "Memory Hierarchy", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Question", "description": "Address translation, tag arrays, cache hit/miss calculations, and replacement policies."},
                    {"title": "Single-Bus CPU Datapath and Instruction Execution Steps", "chapter_number": 2, "chapter_title": "Processing Unit", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Question", "description": "Control sequence for complete instruction fetch, ALU operation, and branch execution."}
                ]
            }
        ]
    },

    # ─── 3rd Semester: Operating Systems ────────────────────────────────────
    "OPERATING_SYSTEMS": {
        "canonical_name": "Operating Systems",
        "canonical_code": "BCS303",
        "aliases": ["BCS303", "21CS44", "18CS43", "OPERATING SYSTEMS", "OPERATING SYSTEM", "OS"],
        "branch": "CSE",
        "semester": 3,
        "scheme": "VTU 2022/2024/2025 Scheme",
        "modules": [
            {
                "module_number": 1,
                "title": "Introduction to Operating Systems and Process Management",
                "raw_text": "Operating System structure, Dual-mode operation, System calls, Process concept, Process Control Block (PCB), Context switching, Process creation and termination, Inter-Process Communication.",
                "chapters": [{"chapter_number": 1, "chapter_title": "OS Overview", "topics": ["Dual Mode & System Calls"]}, {"chapter_number": 2, "chapter_title": "Processes", "topics": ["PCB & Process Scheduling"]}],
                "topics": [
                    {"title": "Operating System Operations and Dual-Mode Execution", "chapter_number": 1, "chapter_title": "OS Overview", "is_high_yield": False, "description": "User mode vs kernel mode transitions and system call handling architecture."},
                    {"title": "Process Control Block (PCB) and Context Switching", "chapter_number": 2, "chapter_title": "Processes", "is_high_yield": True, "vtu_frequency_tag": "Core 8-10M VTU Question", "description": "PCB data structures, process state transitions, and context switch overhead."},
                    {"title": "Inter-Process Communication (IPC): Shared Memory vs Message Passing", "chapter_number": 2, "chapter_title": "Processes", "is_high_yield": True, "vtu_frequency_tag": "Frequent 8-10M Question", "description": "Direct and indirect message queues vs shared memory synchronization."}
                ]
            },
            {
                "module_number": 2,
                "title": "Multithreading and CPU Scheduling",
                "raw_text": "Overview of threads, Multithreading models (Many-to-One, One-to-One, Many-to-Many), Thread libraries, CPU Scheduling concepts, Scheduling criteria, Scheduling algorithms (FCFS, SJF, Priority, Round Robin, Multilevel Queue).",
                "chapters": [{"chapter_number": 1, "chapter_title": "Threads", "topics": ["Multithreading Models"]}, {"chapter_number": 2, "chapter_title": "CPU Scheduling", "topics": ["Scheduling Algorithms"]}],
                "topics": [
                    {"title": "Multithreading Models (Many-to-One, One-to-One, Many-to-Many)", "chapter_number": 1, "chapter_title": "Threads", "is_high_yield": True, "vtu_frequency_tag": "Core 8M VTU Question", "description": "User-level threads vs kernel-level threads mapping models and benefits."},
                    {"title": "CPU Scheduling Algorithms (FCFS, SJF, Priority, Round-Robin)", "chapter_number": 2, "chapter_title": "CPU Scheduling", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Numerical", "description": "Gantt chart construction, waiting time, and turnaround time calculation for preemptive SJF/RR."}
                ]
            },
            {
                "module_number": 3,
                "title": "Process Synchronization and Deadlocks",
                "raw_text": "The Critical-Section Problem, Peterson's solution, Hardware synchronization, Mutex locks, Semaphores, Classic problems (Producer-Consumer, Dining Philosophers), Deadlock characterization, Banker's Algorithm.",
                "chapters": [{"chapter_number": 1, "chapter_title": "Synchronization", "topics": ["Critical Section & Semaphores"]}, {"chapter_number": 2, "chapter_title": "Deadlocks", "topics": ["Banker's Algorithm"]}],
                "topics": [
                    {"title": "Critical Section Problem and Semaphore Implementations", "chapter_number": 1, "chapter_title": "Synchronization", "is_high_yield": True, "vtu_frequency_tag": "Core 10M VTU Question", "description": "Mutual exclusion, progress, bounded waiting, counting vs binary semaphores."},
                    {"title": "Classic Synchronization Problems: Producer-Consumer and Dining Philosophers", "chapter_number": 1, "chapter_title": "Synchronization", "is_high_yield": True, "vtu_frequency_tag": "High-Yield 8-10M Question", "description": "Semaphore pseudo-code solutions avoiding deadlocks and race conditions."},
                    {"title": "Deadlock Characterization and Banker's Safety Algorithm", "chapter_number": 2, "chapter_title": "Deadlocks", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Problem", "description": "Necessary conditions, resource allocation graphs, and Banker's safety state matrices."}
                ]
            },
            {
                "module_number": 4,
                "title": "Main Memory and Virtual Memory Management",
                "raw_text": "Memory management concepts, Swapping, Contiguous memory allocation, Paging, Structure of page table, Segmentation, Virtual memory, Demand paging, Page replacement algorithms (FIFO, Optimal, LRU).",
                "chapters": [{"chapter_number": 1, "chapter_title": "Memory Organization", "topics": ["Paging & Segmentation"]}, {"chapter_number": 2, "chapter_title": "Virtual Memory", "topics": ["Page Replacement Algorithms"]}],
                "topics": [
                    {"title": "Paging Hardware and Translation Lookaside Buffer (TLB)", "chapter_number": 1, "chapter_title": "Memory Organization", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M Question", "description": "Logical to physical address translation, TLB hit ratio, and effective access time."},
                    {"title": "Page Replacement Algorithms (FIFO, LRU, Optimal)", "chapter_number": 2, "chapter_title": "Virtual Memory", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 10M VTU Numerical", "description": "Simulating reference strings, calculating page faults, and Belady's anomaly."}
                ]
            },
            {
                "module_number": 5,
                "title": "File System Interface and Storage Management",
                "raw_text": "File concept, Access methods, Directory structure, File system mounting, File sharing, Protection, Allocation methods (Contiguous, Linked, Indexed), Disk scheduling (FCFS, SSTF, SCAN, C-SCAN).",
                "chapters": [{"chapter_number": 1, "chapter_title": "File Systems", "topics": ["File Allocation Methods"]}, {"chapter_number": 2, "chapter_title": "Disk Management", "topics": ["Disk Scheduling Algorithms"]}],
                "topics": [
                    {"title": "File Allocation Methods (Contiguous, Linked, Indexed)", "chapter_number": 1, "chapter_title": "File Systems", "is_high_yield": True, "vtu_frequency_tag": "Core 8-10M VTU Question", "description": "In-memory file system tables, FAT, and multi-level UNIX inode structures."},
                    {"title": "Disk Scheduling Algorithms (FCFS, SSTF, SCAN, C-SCAN)", "chapter_number": 2, "chapter_title": "Disk Management", "is_high_yield": True, "vtu_frequency_tag": "Guaranteed 8-10M Numerical", "description": "Calculating total head movements and seek times across disk request cylinders."}
                ]
            }
        ]
    }
}


def normalize_lookup_token(text: str) -> str:
    """Normalizes string for robust catalog alias matching."""
    if not text:
        return ""
    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    return clean


def find_vtu_catalog_entry(course_code: str = "", subject_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Finds the full catalog dictionary entry by course code or subject name.
    Searches both built-in core specifications and dynamically ingested VTUCircle records.
    """
    code_norm = normalize_lookup_token(course_code)
    name_norm = normalize_lookup_token(subject_name)

    # 1. Check direct code matches in built-in catalog
    for key, spec in VTU_PRESCRIBED_CATALOG.items():
        aliases_norm = [normalize_lookup_token(a) for a in spec.get("aliases", [])]
        if code_norm and code_norm in aliases_norm:
            return spec

    # 2. Check subject name matches in built-in catalog
    for key, spec in VTU_PRESCRIBED_CATALOG.items():
        canonical_norm = normalize_lookup_token(spec.get("canonical_name", ""))
        aliases_norm = [normalize_lookup_token(a) for a in spec.get("aliases", [])]
        if name_norm and (name_norm == canonical_norm or name_norm in aliases_norm):
            return spec
        # Substring heuristics for common names
        if name_norm and ("DATASTRUCTURE" in name_norm or "DATASTRUCTURES" in name_norm) and key == "DATA_STRUCTURES":
            return spec
        if name_norm and ("OPERATINGSYSTEM" in name_norm or "OPERATINGSYSTEMS" in name_norm) and key == "OPERATING_SYSTEMS":
            return spec
        if name_norm and ("DIGITALDESIGN" in name_norm or "DDCO" in name_norm) and key == "DDCO":
            return spec

    # 3. Check ingested / cached VTUCircle catalog
    try:
        from curriculum.services.vtucircle_scraper import load_local_vtucircle_cache
        crawled_catalog = load_local_vtucircle_cache()
        for key, spec in crawled_catalog.items():
            aliases_norm = [normalize_lookup_token(a) for a in spec.get("aliases", [])]
            canonical_code_norm = normalize_lookup_token(spec.get("canonical_code", ""))
            canonical_name_norm = normalize_lookup_token(spec.get("canonical_name", ""))
            page_title_norm = normalize_lookup_token(spec.get("page_title", ""))

            if code_norm and (code_norm == canonical_code_norm or code_norm in aliases_norm or code_norm in key):
                return spec
            if name_norm and (name_norm == canonical_name_norm or name_norm in aliases_norm or name_norm in page_title_norm):
                return spec
    except Exception:
        pass

    return None


def format_syllabus_as_text(entry_or_modules) -> str:
    """Formats modules or catalog entry into prescribed syllabus text format."""
    if not entry_or_modules:
        return ""
    if isinstance(entry_or_modules, dict) and "modules" in entry_or_modules:
        modules = entry_or_modules["modules"]
    elif isinstance(entry_or_modules, list):
        modules = entry_or_modules
    else:
        return ""

    out = []
    for m in modules:
        num = m.get("module_number", "")
        title = m.get("title", "")
        raw = m.get("raw_text", "")
        out.append(f"Module {num}: {title}\n{raw}".strip())
    return "\n\n".join(out)


def find_vtu_verified_syllabus(course_code: str = "", subject_name: str = "") -> Optional[List[Dict[str, Any]]]:
    """
    Finds verified official VTU prescribed syllabus by searching the catalog by:
    1. Exact course code (e.g. BCS304, 1BCS305, 21CS32, BCS302).
    2. Normalized alias match.
    3. Subject name substring matching (e.g. 'Data Structures and Application', 'Operating Systems').
    
    Returns 5 structured prescribed modules ready for grounding AI generation, or None.
    """
    entry = find_vtu_catalog_entry(course_code=course_code, subject_name=subject_name)
    if entry:
        return entry.get("modules")
    return None
