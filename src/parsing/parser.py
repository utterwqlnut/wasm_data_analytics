def parse(s=""):
    parts = []
    commands = []
    literal_mode=False
    curr = ""
    count = 0
    # Split up syntax and be whitespace robust, account for multiple edge cases
    for i in range(len(s)):
        if s[i] == '"':
            literal_mode = not literal_mode

        if s[i]==' ' and not literal_mode:
            continue
        
        if s[i]=='(':
            parts.append(curr+'_'+str(len(parts)))
            parts.append(s[i])
            curr=""
        elif s[i]==')':
            if '\"' in curr:
                parts.append(curr+'_'+str(len(parts)))
            parts.append(s[i])
            curr=""
        elif s[i]==',':
            if curr!="":
                parts.append(curr+'_'+str(len(parts)))
                curr = ""
        else:
            curr+=s[i]


    for i in range(len(parts)):
        if parts[i]!='(' and parts[i]!=')':
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

print(parse("plot(line(\"Year\",\"Value\"))"))