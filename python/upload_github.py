#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:56:43 2020

@author: kimondrakopoulos
"""


from github import Github
from github import InputGitTreeElement

user = "ovelixasterix"
password = "s35653565!"

g = Github("52483470529ec0e21a240577417719dd9d3000ec")

g=Github("ovelixasterix","s35653565!")
  
repo = g.get_user().get_repo("rnn")

all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))




#-----Upload future.csv=====================
with open('../tmp/csv/future.csv', 'r') as file:
    content = file.read()

git_file = 'future.csv'

if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, "committing files", content, branch="master")
    print(git_file + ' CREATED')
    
    
#-----Upload future-spain.csv=====================
with open('../tmp/csv/future_spain.csv', 'r') as file:
    content = file.read()

git_file = 'future_spain.csv'


if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, "committing files", content, branch="master")
    print(git_file + ' CREATED')
    
#-----Upload ranking.csv=====================
with open('../tmp/csv/ranking.csv', 'r') as file:
    content = file.read()

git_file = 'ranking.csv'


if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, "committing files", content, branch="master")
    print(git_file + ' CREATED')
    
#-----Upload future_morecountriestest.csv=====================
with open('../tmp/csv/future_morecountriestest.csv', 'r') as file:
    content = file.read()

git_file = 'future_morecountriestest.csv'


if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, "committing files", content, contents.sha, branch="master")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, "committing files", content, branch="master")
    print(git_file + ' CREATED')
   
    
    
    
    
    