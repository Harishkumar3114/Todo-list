import React, { useState } from 'react'
import Todoform from './Todoform'
import {v4 as uuidv4} from 'uuid'
import Todo from './Todo';
import EditTodo from './EditTodo';
uuidv4();

const Todowrapper = () => {
    const [todoarray,Settodos] =useState([])
    const addprops= todo => {
        Settodos([...todoarray,{id:uuidv4(),task:todo,completed:false,isEditing:false}])
        console.log(todoarray)
    }
    const toggleComplete = (id) =>{
         Settodos(todoarray.map(todo=> todo.id===id ? {...todo,completed:!todo.completed} :todo ))
    }

    const deleteTodo= (id) => {
        Settodos(todoarray.filter(todo =>todo.id !== id))
    }

    const editTodo = (id) =>{
        Settodos(todoarray.map(todo => todo.id===id ? {
            ...todo,isEditing: !todo.isEditing} : todo
        ))
    }
    const editTask =(task,id) => {
        Settodos(todoarray.map(todo => todo.id===id ? {
            ...todo,task,isEditing: !todo.isEditing} : todo
        ))
    }
  return (
    <div className='TodoWrapper'>
        <h1>Get things done</h1>
      <Todoform addprops={addprops}/>
      {todoarray.map((todo,index)=>(
        todo.isEditing ? (
            <EditTodo editTodo={editTask} task={todo}/>
        ):(
            <Todo task={todo} key={index} toggleComplete={toggleComplete} deleteTodo={deleteTodo} editTodo={editTodo} />
        )
      
      ))}
    
    </div>
  )
}

export default Todowrapper
