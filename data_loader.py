import os
import pandas as pd
import numpy as np
import io
import config

INSTITUTIONAL_RAW_DATA = """Sl.No,Email,Full Name,USN,Phone,BRANCH,Section,L1,L2,L3,L4,A1,A2,A3,A4,S1,S2,S3,S4,C2_Odd,C2_Full,C3_Odd,C3_Full,C4_Odd,C4_Full,C5_Full,P1_C,P2_Python,P3_Python,P3_Java,P4_Part1,P4_Part2,P4_MAD_FSD,P4_DS,P2_Plus_Python,Interview_Readiness,Lx_Level,Ax_Level,Cx_Level,Px_Level,Sx_Level
1,student01@college.edu,Aarav Sharma,USN23AI001,9876543201,CSE(AIML),A,78,80,72,70,61,50,50,50,81,50,91,50,14,25,92,90,84,96,0,86,100,76,75,75,70,96,77,0,4,4,4,4,4
2,student02@college.edu,Aditya Varma,USN23AI002,9876543202,CSE(AIML),A,76,82,70,70,50,50,50,50,50,67,85,52,14,22,88,82,68,58,0,0,88,82,60,0,33,43,56,0,4,4,4,3,4
3,student03@college.edu,Akshita Nair,USN23AI003,9876543203,CSE(AIML),A,86,87,70,71,50,52,50,50,83,75,50,59,11,20,92,90,66,100,0,86,88,80,90,6,2,88,58,0,4,4,4,3,4
4,student04@college.edu,Ananya Sen,USN23AI004,9876543204,CSE(AIML),A,70,80,70,70,50,54,50,50,77,50,92,50,18,21,88,86,70,100,0,12,100,33,86,70,32,80,77,0,4,4,4,3.5,4
5,student05@college.edu,Aman Bhardwaj,USN23AI005,9876543205,CSE(AIML),A,79,84,36,6,8,0,0,0,80,0,0,0,10,25,0,0,0,0,0,28,57,0,33,10,0,62,20,0,2,0,2,2,1
6,student06@college.edu,Anjali Munagala,USN23AI006,9876543206,CSE(AIML),A,82,85,72,70,50,51,58,50,81,67,88,62,15,24,90,90,74,100,0,79,100,81,65,70,85,33,91,0,4,4,4,4,4
7,student07@college.edu,Ankit Singh,USN23AI007,9876543207,CSE(AIML),A,80,82,79,77,59,55,60,57,82,72,88,71,13,21,86,90,78,98,0,90,100,77,60,84,70,85,85,0,4,4,4,4,4
8,student08@college.edu,Arpita Roy,USN23AI008,9876543208,CSE(AIML),A,66,72,13,13,24,50,13,42,80,62,0,30,11,22,92,86,78,68,0,58,47,0,0,10,0,56,38,0,2,0,4,0,2
9,student09@college.edu,Arya Patil,USN23AI009,9876543209,CSE(AIML),A,71,80,70,70,50,50,50,50,80,50,50,50,16,19,88,90,78,98,0,92,79,57,60,70,75,30,70,0,4,4,4,4,4
10,student10@college.edu,Aryan Srivastava,USN23AI010,9876543210,CSE(AIML),A,80,92,71,76,50,50,50,50,80,62,94,78,11,20,90,90,76,100,0,58,100,70,86,70,70,83,42,0,4,4,4,4,4
11,student11@college.edu,Bhavin Kumar,USN23AI011,9876543211,CSE(AIML),A,76,75,68,70,50,50,17,24,50,50,50,0,17,20,86,90,68,100,0,23,20,44,1,70,0,33,40,0,2,2,4,0,3
12,student12@college.edu,Bhavana Sharma,USN23AI012,9876543212,CSE(AIML),A,87,98,81,83,73,56,59,56,82,73,94,82,19,21,90,90,78,100,0,74,100,98,65,95,77,99,90,0,4,4,4,4,4
13,student13@college.edu,Dhanush Gowda,USN23AI013,9876543213,CSE(AIML),A,70,86,76,70,50,54,52,50,50,67,94,50,13,24,90,90,72,100,0,84,100,97,42,75,70,50,83,0,4,4,4,4,4
14,student14@college.edu,Farhan Raza,USN23AI014,9876543214,CSE(AIML),A,70,76,70,70,50,50,50,50,83,90,80,50,21,25,92,90,68,100,0,8,57,84,60,10,88,60,38,0,4,4,4,3.5,4
15,student15@college.edu,Gagan Kiran,USN23AI015,9876543215,CSE(AIML),A,70,78,70,70,50,50,50,50,73,50,91,61,12,24,92,90,70,100,0,54,71,62,60,31,70,70,58,0,4,4,4,3.5,4
16,student16@college.edu,Gayatri Rao,USN23AI016,9876543216,CSE(AIML),A,71,92,86,75,50,52,50,50,80,70,91,71,16,22,92,90,80,98,0,62,93,63,85,77,70,76,71,0,4,4,4,4,4
17,student17@college.edu,Hema Challa,USN23AI017,9876543217,CSE(AIML),A,83,89,70,71,50,55,61,58,80,75,93,53,14,24,92,90,64,100,0,79,97,62,60,75,70,78,77,0,4,4,4,4,4
18,student18@college.edu,Ishan Chandra,USN23AI018,9876543218,CSE(AIML),A,89,86,70,82,50,50,50,64,76,53,94,55,14,24,90,88,76,96,0,50,100,78,90,75,70,82,39,0,4,4,4,4,4
19,student19@college.edu,Yashwanth Kumar,USN23AI019,9876543219,CSE(AIML),A,76,83,78,70,50,50,50,50,83,90,93,56,13,24,88,90,80,100,0,92,91,82,60,94,96,82,74,0,4,4,4,4,4
20,student20@college.edu,Bhuvan Sriram,USN23AI020,9876543220,CSE(AIML),A,72,86,70,70,50,51,50,50,73,50,90,50,13,22,92,90,76,98,0,34,70,100,80,60,60,38,33,0,4,4,4,3,4
21,student21@college.edu,Karthik Nair,USN23AI021,9876543221,CSE(AIML),A,73,86,80,70,62,55,50,63,50,67,91,65,15,22,92,90,84,94,0,80,100,99,62,76,92,0,82,0,4,4,4,4,4
22,student22@college.edu,Venkat Danush,USN23AI022,9876543222,CSE(AIML),A,79,86,87,78,74,61,52,66,76,71,91,67,12,25,92,86,70,100,0,96,100,89,80,75,79,98,91,0,4,4,4,4,4
23,student23@college.edu,Neha Gaikwad,USN23AI023,9876543223,CSE(AIML),A,68,83,69,12,50,17,5,0,81,0,34,0,15,25,92,90,0,0,0,74,92,81,0,25,0,0,44,0,2,1,3,3,1
24,student24@college.edu,Deepika Reddy,USN23AI024,9876543224,CSE(AIML),A,85,96,87,87,50,65,72,68,83,70,92,75,15,21,88,90,76,98,0,82,100,94,100,70,70,83,72,0,4,4,4,4,4
25,student25@college.edu,Chakradhar Reddy,USN23AI025,9876543225,CSE(AIML),A,86,92,70,73,50,50,50,54,83,50,50,64,13,25,92,88,70,100,0,92,97,100,13,76,80,89,87,0,4,4,4,4,4
26,student26@college.edu,Manya Michelle,USN23AI026,9876543226,CSE(AIML),A,73,77,79,71,50,50,50,56,82,72,87,63,10,22,88,90,80,100,0,100,100,96,75,100,85,85,73,0,4,4,4,4,4
27,student27@college.edu,Zaid Ibrahim,USN23AI027,9876543227,CSE(AIML),A,79,88,70,70,50,50,50,50,50,90,81,50,10,25,88,90,84,98,0,30,50,70,60,71,94,57,32,0,4,4,4,3.5,4
28,student28@college.edu,Meghana Ganta,USN23AI028,9876543228,CSE(AIML),A,80,90,70,70,50,50,60,69,77,50,83,50,10,20,94,90,66,100,0,58,94,62,80,71,70,73,48,0,4,4,4,4,4
29,student29@college.edu,Naveen Kumar,USN23AI029,9876543229,CSE(AIML),A,81,83,76,70,50,50,50,58,73,71,94,50,14,25,92,84,86,100,0,100,100,84,60,77,77,94,88,0,4,4,4,4,4
30,student30@college.edu,Naman Jain,USN23AI030,9876543230,CSE(AIML),A,74,90,70,70,50,0,0,0,0,0,0,0,12,25,90,90,0,0,0,60,95,0,2,10,0,0,33,0,4,1,3,2,0
31,student31@college.edu,Nayana Rao,USN23AI031,9876543231,CSE(AIML),A,74,85,84,81,50,64,52,71,77,62,90,77,20,22,92,88,72,100,0,92,90,99,60,76,77,97,85,0,4,4,4,4,4
32,student32@college.edu,Nikita Prabhu,USN23AI032,9876543232,CSE(AIML),A,89,89,70,81,54,54,50,56,79,67,82,50,15,17,78,90,72,100,0,100,91,73,80,75,61,85,90,0,4,4,4,3.5,4
33,student33@college.edu,Amarnath Reddy,USN23AI033,9876543233,CSE(AIML),A,70,75,70,70,50,50,50,50,84,50,50,50,22,25,86,90,80,100,0,34,95,43,90,92,45,0,39,0,4,4,4,3.5,4
34,student34@college.edu,Pallavi Kumbar,USN23AI034,9876543234,CSE(AIML),A,82,90,71,70,78,53,50,70,81,72,87,77,17,22,92,90,76,100,0,94,100,75,65,75,78,65,88,0,4,4,4,4,4
35,student35@college.edu,Meghana Patil,USN23AI035,9876543235,CSE(AIML),A,80,86,70,70,62,50,50,50,73,90,89,50,12,24,92,88,64,100,0,77.6,88,28,86,17,61,88,84,0,4,4,4,3,4
36,student36@college.edu,Manikanta P,USN23AI036,9876543236,CSE(AIML),A,70,77,70,70,50,50,50,50,50,50,94,50,15,22,92,86,82,98,0,20,50,80,90,82,94,32,34,0,4,4,4,3.5,4
37,student37@college.edu,Preethi Nair,USN23AI037,9876543237,CSE(AIML),A,82,88,80,70,52,50,50,50,82,75,67,71,14,25,92,90,84,100,0,74,92,26,80,75,73,98,80,0,4,4,4,4,4
38,student38@college.edu,Priyanshu Madhup,USN23AI038,9876543238,CSE(AIML),A,83,90,71,70,59,54,50,59,82,59,87,75,18,25,92,90,82,98,0,100,75,86,60,76,91,98,91,0,4,4,4,4,4
39,student39@college.edu,Purvi K,USN23AI039,9876543239,CSE(AIML),A,87,89,75,70,58,50,52,55,80,62,94,78,14,24,90,92,66,86,0,84,92,69,60,70,85,98,87,0,4,4,4,4,4
40,student40@college.edu,Rachit Joseph,USN23AI040,9876543240,CSE(AIML),A,87,96,86,60,41,12,6,0,84,71,0,0,18,24,90,90,82,100,0,96,100,50,48,10,0,25,70,0,3,0,4,2,2
41,student41@college.edu,Ramya S,USN23AI041,9876543241,CSE(AIML),A,74,83,70,70,50,50,53,50,78,62,93,61,19,24,88,90,72,100,0,60,96,65,60,70,70,98,82,0,4,4,4,4,4
42,student42@college.edu,Param Raval,USN23AI042,9876543242,CSE(AIML),A,70,70,70,70,50,50,57,50,83,50,93,64,13,25,92,90,76,100,0,76,88,68,46,73,85,86,88,0,4,4,4,4,4
43,student43@college.edu,Rithika Voona,USN23AI043,9876543243,CSE(AIML),A,88,94,77,70,65,59,56,58,80,70,88,60,16,21,92,90,78,100,0,88,100,79,100,74,70,88,85,0,4,4,4,4,4
44,student44@college.edu,Rohan Reddy,USN23AI044,9876543244,CSE(AIML),A,67,83,65,4,11,0,0,0,0,0,0,0,14,15,66,84,0,0,0,94,0,0,53,10,0,0,26,0,2,0,3,0,0
45,student45@college.edu,Samarth Hegde,USN23AI045,9876543245,CSE(AIML),A,72,70,70,70,65,67,50,58,50,50,87,51,12,23,88,86,72,96,0,62,100,100,60,75,76,92,78,0,4,4,4,4,4
46,student46@college.edu,Sanketh R,USN23AI046,9876543246,CSE(AIML),A,70,74,70,70,50,59,50,52,78,71,87,64,11,25,92,86,74,100,0,60,100,100,60,75,70,94,81,0,4,4,4,4,4
47,student47@college.edu,Chanukya N,USN23AI047,9876543247,CSE(AIML),A,70,79,70,70,50,50,50,50,83,50,81,50,14,24,90,86,70,100,0,68,81,90,60,100,70,32,36,0,4,4,4,3.5,4
48,student48@college.edu,Shaan Poonacha,USN23AI048,9876543248,CSE(AIML),A,71,84,30,4,0,0,0,0,0,0,0,0,13,12,90,90,70,94,0,24,0,0,1,10,0,0,15,0,2,0,4,0,0
49,student49@college.edu,Shaiza Samreen,USN23AI049,9876543249,CSE(AIML),A,84,93,91,84,50,50,53,54,78,75,69,76,13,25,92,88,70,100,0,60.8,96,34,60,70,72,98,91,0,4,4,4,4,4
50,student50@college.edu,Sharat Patil,USN23AI050,9876543250,CSE(AIML),A,77,74,73,70,50,50,50,50,50,71,81,50,15,25,90,90,68,96,0,84,95,79,75,80,86,75,48,0,4,4,4,4,4
51,student51@college.edu,Shreya Dhange,USN23AI051,9876543251,CSE(AIML),A,81,93,73,70,61,50,53,61,81,72,86,78,14,24,94,90,84,100,0,84,88,100,85,70,70,70,62,0,4,4,4,4,4
52,student52@college.edu,Shriya Nair,USN23AI052,9876543252,CSE(AIML),A,83,83,60,10,39,0,12,0,81,0,0,0,21,25,92,90,0,100,0,34,73,0,1,15,0,66,27,0,2,0,4,2,1
53,student53@college.edu,Smruti Rao,USN23AI053,9876543253,CSE(AIML),A,91,98,80,79,56,50,50,61,78,62,92,62,14,24,92,90,78,100,0,92,100,83,60,72,94,100,95,0,4,4,4,4,4
54,student54@college.edu,Sreethu S,USN23AI054,9876543254,CSE(AIML),A,85,94,85,73,54,50,50,50,81,75,90,74,20,24,86,90,86,100,0,100,100,32,60,70,84,100,87,0,4,4,4,4,4
55,student55@college.edu,Sudhanshu Shetty,USN23AI055,9876543255,CSE(AIML),A,79,70,70,70,50,50,50,50,77,67,50,50,21,22,82,90,78,100,0,62,50,57,60,72,36,27,50,0,4,4,4,3.5,4
56,student56@college.edu,Tanushree A,USN23AI056,9876543256,CSE(AIML),A,90,92,81,74,54,50,55,50,80,75,95,59,21,25,90,90,84,100,0,60.8,100,70,60,70,70,98,84,0,4,4,4,4,4
57,student57@college.edu,Keerthana V,USN23AI057,9876543257,CSE(AIML),A,78,87,70,70,50,50,50,50,83,90,50,50,19,25,92,88,84,100,0,18,77,4,86,70,8,82,70,0,4,4,4,3.5,4
58,student58@college.edu,Vaishnav S,USN23AI058,9876543258,CSE(AIML),A,68,77,50,4,8,15,20,0,50,0,0,0,16,20,82,0,76,0,0,16,21,0,0,10,0,53,17,0,2,0,2,0,1
59,student59@college.edu,Varshini S,USN23AI059,9876543259,CSE(AIML),A,80,87,70,70,50,50,50,38,79,70,52,50,11,21,92,80,78,100,0,74,100,90,37,19,34,80,48,0,4,3,4,3,4
60,student60@college.edu,Vishnu Menon,USN23AI060,9876543260,CSE(AIML),A,86,89,84,75,59,50,50,50,80,71,86,69,14,25,92,86,86,100,0,100,75,99,60,70,70,89,95,0,4,4,4,4,4
61,student61@college.edu,Yash Rajput,USN23AI061,9876543261,CSE(AIML),A,80,88,79,70,54,50,50,50,50,67,89,55,13,23,92,88,68,100,0,74,97,22,60,36,95,86,82,0,4,4,4,3.5,4
62,student62@college.edu,Aishwarya J,USN23AI062,9876543262,CSE(AIML),A,16,24,36,70,50,50,-4,0,0,0,0,0,15,15,92,90,68,96,0,0,41,41,42,10,0,33,35,0,0,2,4,0,0
63,student63@college.edu,Harshavardhan Reddy,USN23AI063,9876543263,CSE(AIML),A,40,28,39,18,7,12,1,2,50,50,50,0,15,15,88,88,68,96,0,0,3,3,42,10,0,31,24,0,0,0,4,0,3
64,student64@college.edu,Nagaraj H,USN23AI064,9876543264,CSE(AIML),A,44,32,48,23,0,14,3,6,50,0,50,0,15,15,92,86,74,100,0,0,1,1,0,10,0,38,24,0,0,0,4,0,1
65,student65@college.edu,Puneeth D,USN23AI065,9876543265,CSE(AIML),A,65,65,70,70,50,50,19,15,50,71,50,0,15,15,92,86,70,98,0,0,49,49,80,15,0,4,66,0,4,2,4,0,3
66,student66@college.edu,Sandesh Naik,USN23AI066,9876543266,CSE(AIML),A,65,65,70,70,17,50,50,16,50,50,68,50,15,15,92,90,68,100,0,0,100,100,70,44,16,4,67,0,4,0,4,3,4
67,student67@college.edu,Nilotpal Arya,USN23AI067,9876543267,CSE(AIML),A,94,92,82,72,56,56,55,66,78,72,87,74,18,17,62,90,74,100,0,0,100,65,70,70,96,65,84,0,4,4,4,4,4
"""

CMRIT_RAW_DATA = INSTITUTIONAL_RAW_DATA

def generate_placement_dataset(n_samples=500, random_seed=42):
    """
    Generates realistic synthetic placement dataset corresponding to the exact 
    profile described in the project report (500 students, 337 placed [67.4%], 163 not placed [32.6%]).
    """
    np.random.seed(random_seed)
    n_placed = 337
    n_not_placed = 163

    # Generate Placed cohort to match Table 3.2:
    # Means: CGPA 7.84, Aptitude 73.92, Comm 71.71, Tech 74.54, Internships 1.47, Projects 2.81, Certs 1.85, Backlogs 0.74
    placed_cgpa = np.clip(np.random.normal(7.84, 0.65, n_placed), 6.0, 9.9)
    placed_apt = np.clip(np.random.normal(73.92, 10.5, n_placed), 45, 99)
    placed_comm = np.clip(np.random.normal(71.71, 9.8, n_placed), 48, 98)
    placed_tech = np.clip(np.random.normal(74.54, 11.2, n_placed), 45, 100)
    placed_intern = np.clip(np.random.poisson(1.47, n_placed), 0, 5)
    placed_proj = np.clip(np.random.poisson(2.81, n_placed), 1, 7)
    placed_certs = np.clip(np.random.poisson(1.85, n_placed), 0, 6)
    placed_backlogs = np.clip(np.random.poisson(0.74, n_placed), 0, 4)
    placed_target = np.ones(n_placed, dtype=int)

    # Generate Not-Placed cohort to match Table 3.2:
    # Means: CGPA 6.84, Aptitude 58.74, Comm 60.27, Tech 60.38, Internships 0.75, Projects 1.66, Certs 1.06, Backlogs 1.66
    not_cgpa = np.clip(np.random.normal(6.84, 0.72, n_not_placed), 5.0, 8.4)
    not_apt = np.clip(np.random.normal(58.74, 12.0, n_not_placed), 20, 85)
    not_comm = np.clip(np.random.normal(60.27, 11.5, n_not_placed), 25, 82)
    not_tech = np.clip(np.random.normal(60.38, 12.8, n_not_placed), 20, 86)
    not_intern = np.clip(np.random.poisson(0.75, n_not_placed), 0, 3)
    not_proj = np.clip(np.random.poisson(1.66, n_not_placed), 0, 4)
    not_certs = np.clip(np.random.poisson(1.06, n_not_placed), 0, 4)
    not_backlogs = np.clip(np.random.poisson(1.66, n_not_placed), 0, 6)
    not_target = np.zeros(n_not_placed, dtype=int)

    # Combine
    cgpa = np.concatenate([placed_cgpa, not_cgpa])
    aptitude = np.concatenate([placed_apt, not_apt])
    comm = np.concatenate([placed_comm, not_comm])
    tech = np.concatenate([placed_tech, not_tech])
    intern = np.concatenate([placed_intern, not_intern])
    proj = np.concatenate([placed_proj, not_proj])
    certs = np.concatenate([placed_certs, not_certs])
    backlogs = np.concatenate([placed_backlogs, not_backlogs])
    target = np.concatenate([placed_target, not_target])

    student_ids = [f"STU{i+1:04d}" for i in range(n_samples)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "cgpa": np.round(cgpa, 2),
        "aptitude_score": np.round(aptitude, 1),
        "communication_score": np.round(comm, 1),
        "technical_score": np.round(tech, 1),
        "internships": intern.astype(int),
        "projects": proj.astype(int),
        "certifications": certs.astype(int),
        "backlogs": backlogs.astype(int),
        "placed": target.astype(int)
    })

    # Shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def ensure_dataset_files():
    """Ensures data directory and CSV files exist."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    if not os.path.exists(config.DATASET_PATH):
        df = generate_placement_dataset(500, config.RANDOM_STATE)
        df.to_csv(config.DATASET_PATH, index=False)
        print(f"[+] Created synthetic placement dataset at: {config.DATASET_PATH}")
    
    if not os.path.exists(config.INSTITUTIONAL_DATA_PATH):
        df_inst = pd.read_csv(io.StringIO(INSTITUTIONAL_RAW_DATA))
        df_inst.to_csv(config.INSTITUTIONAL_DATA_PATH, index=False)
        print(f"[+] Saved institutional assessment dataset at: {config.INSTITUTIONAL_DATA_PATH}")

def load_data():
    """Loads student placement dataset."""
    ensure_dataset_files()
    return pd.read_csv(config.DATASET_PATH)

def load_institutional_data():
    """Loads institutional assessment records."""
    ensure_dataset_files()
    return pd.read_csv(config.INSTITUTIONAL_DATA_PATH)

def load_cmrit_data():
    """Alias for load_institutional_data."""
    return load_institutional_data()

if __name__ == "__main__":
    ensure_dataset_files()
    df = load_data()
    print("Placement Dataset Sample:")
    print(df.head())
    print("\nDataset Class Distribution:")
    print(df["placed"].value_counts(normalize=True))
    print("\nInstitutional Assessment Sample:")
    print(load_institutional_data().head())
