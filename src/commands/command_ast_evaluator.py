from command_builder import Builder

class AstEvaluator:
    def __init__(self,builder: Builder):
        self.builder = builder

    def ast_evaluator_main(self,ast):
        initial_key = ast.keys()[0]
        return self.ast_evaluator_loop(ast,initial_key)
    
    # Simple DFS over AST
    def ast_evaluator_loop(self,ast,key):
        # If a command has all literal parameters
        if all(len(ast[grandchild])==0 for grandchild in ast[key]):
            return self.builder.build(key,ast[key]) # Command and its parameters
        
        for child in ast[key]:
            parameters = []
            parameters.append(AstEvaluator.ast_evaluator_loop(ast,child))

            return self.builder.build(key,parameters)