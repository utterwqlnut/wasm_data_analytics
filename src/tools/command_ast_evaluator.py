from .command_builder import Builder
import streamlit as st

class AstEvaluator:
    def __init__(self,builder: Builder):
        self.builder = builder

    def ast_evaluator_main(self,ast):
        initial_key = list(ast.keys())[0]
        return self.ast_evaluator_loop(ast,initial_key)
    
    # Simple DFS over AST
    def ast_evaluator_loop(self,ast,key):
        # If a command is a literal
        if len(ast[key])==0:
            return key # Just return itself
        
        parameters = []
        for child in ast[key]:
            parameters.append(self.ast_evaluator_loop(ast,child))

        return self.builder.build(key,parameters)