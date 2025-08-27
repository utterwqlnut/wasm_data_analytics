def parse(s=""):
    parts = []
    commands = []
    literal_mode=False
    curr = ""
    # Split up syntax and be whitespace robust, account for multiple edge cases
    for i in range(len(s)):
        if s[i]=="\"" and literal_mode==False:
            literal_mode=True
        elif s[i]=="\"" and literal_mode==True:
            literal_mode=False

        if s[i]==' ' and not literal_mode:
            continue
        if s[i]=='(':
            parts.append(curr)
            parts.append(s[i])
            curr=""
        elif s[i]==')':
            if '\"' in curr:
                parts.append(curr)
            parts.append(s[i])
            curr=""
        elif s[i]==',':
            if curr!="":
                parts.append(curr)
                curr = ""
        elif s[i]!=' ':
            curr+=s[i]

    for i in range(len(parts)):
        if parts[i]!='(' and parts[i]!=')' and "\"" not in parts[i]:
            commands.append(parts[i])
    
    # Build mini abstract syntax tree in adjacency list format via a stack

    # Initialize adjacency list
    adj_list = {}
    for i in range(len(commands)):
        adj_list[commands[i]] = []

    stack_outer = []

    for i in range(len(parts)):
        # Skip first keyword
        if i==0:
            continue

        if parts[i]=='(':
            stack_outer.append(parts[i-1])
        
        elif parts[i]==')':
            stack_outer.pop()
        else:
            adj_list[stack_outer[-1]].append(parts[i])


    return adj_list